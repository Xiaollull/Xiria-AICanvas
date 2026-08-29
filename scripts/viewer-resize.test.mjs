import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { inverseViewerHandleScale, viewerResizeDisableDecision } from "../src/viewer-geometry.js";

test("inverse total transform keeps overlay visual and hit sizes screen-fixed", () => {
  for (const [zoom, scale] of [[.05, .1], [1, 1], [32, 8]]) {
    const inverse = inverseViewerHandleScale(zoom, scale);
    assert.ok(Math.abs(13 * zoom * scale * inverse - 13) < 1e-10);
    assert.ok(Math.abs(20 * zoom * scale * inverse - 20) < 1e-10);
  }
  assert.equal(inverseViewerHandleScale(Infinity, NaN), 200);
});

test("resize UI is default-on, resets on open, and every layer scale entry fails closed", () => {
  const source = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(source, /useState\(true\).*viewerLayerResizeEnabled|viewerLayerResizeEnabled.*useState\(true\)/s);
  assert.match(source, /openImageViewer[\s\S]{0,900}setViewerLayerResizeEnabled\(true\)/);
  assert.match(source, /const scaleViewerLayer[\s\S]{0,120}if \(!viewerLayerResizeEnabled\) return/);
  assert.match(source, /const setViewerLayerScale[\s\S]{0,120}if \(!viewerLayerResizeEnabled\) return/);
  assert.match(source, /resizeHandle && !viewerLayerResizeEnabled/);
  assert.match(source, /viewerLayerResizeEnabled && activeViewerLayer === layer\.id/);
  assert.match(source, /disabled=\{!viewerLayerResizeEnabled\}/);
  assert.match(source, /aria-pressed=\{viewerLayerResizeEnabled\}/);
  assert.match(source, /finishViewerResizeForDisable/);
  assert.match(source, /inverseViewerHandleScale\(viewerZoom, layer\.scale\)/);
  assert.match(source, /releasePointerCapture/);
  assert.match(source, /viewerResizeDisableDecision\(drag\)/);
  assert.match(source, /viewerResizeGestureMove\(drag\.resizeGesture, nextTransform\)/);
  assert.match(source, /drag\.changed = drag\.resizeGesture\.changed/);
  assert.match(source, /if \(drag\.resizeGesture\.shouldApply\)[\s\S]{0,300}updateViewerLayer\(drag\.id, nextTransform\)/);
});

test("Viewer DOM/CSS contract has eight anchored fixed-screen handles; browser pixels remain a manual check", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  const css = fs.readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
  assert.match(app, /\["tl", "tr", "bl", "br", "top", "right", "bottom", "left"\]\.map/);
  for (const anchor of ["tl", "tr", "bl", "br", "top", "right", "bottom", "left"]) assert.match(css, new RegExp(`\\.layer-corner-anchor\\.${anchor}`));
  assert.match(css, /width: 20px; height: 20px/);
  assert.match(css, /width: 13px; height: 13px/);
  assert.match(css, /transform: scale\(var\(--viewer-handle-inverse\)\)/);
});

test("disable cleanup is idempotent after its decision has cleared the resize drag", () => {
  const changed = viewerResizeDisableDecision({ kind: "resize", changed: true, pointerId: 9 });
  const afterClear = viewerResizeDisableDecision(null);
  assert.equal(changed.shouldSaveUndo, true);
  assert.equal(afterClear.shouldSaveUndo, false);
  assert.equal(afterClear.shouldFinish, false);
});
