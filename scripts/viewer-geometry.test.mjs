import test from "node:test";
import assert from "node:assert/strict";
import { fitViewerZoom, inverseViewerHandleScale, intrinsicDimensions, viewerHandleScreenMetrics, viewerResizeChanged, viewerResizeDisableDecision, viewerResizeGestureFinish, viewerResizeGestureMove, viewerSceneBounds, viewerZoomAtPoint, VIEWER_HANDLE_HIT_PX, VIEWER_HANDLE_VISUAL_PX, VIEWER_MAX_INVERSE_SCALE, VIEWER_MIN_INVERSE_SCALE } from "../src/viewer-geometry.js";

const layer = (w, h, x = 0, y = 0, scale = 1) => ({ naturalWidth: w, naturalHeight: h, x, y, scale });

test("each layer keeps its own intrinsic pixels and source-independent scale", () => {
  const first = layer(3840, 2160);
  const second = layer(1024, 768, 200, 100);
  assert.deepEqual(intrinsicDimensions(first.naturalWidth, first.naturalHeight), { naturalWidth: 3840, naturalHeight: 2160 });
  assert.deepEqual(intrinsicDimensions(second.naturalWidth, second.naturalHeight), { naturalWidth: 1024, naturalHeight: 768 });
  assert.equal(first.scale, second.scale);
  assert.notDeepEqual(first, second);
});

test("fit is order-independent, 100% is source-pixel scale, and does not mutate sources", () => {
  const layers = [layer(3840, 2160), layer(1024, 768, 2200, 900)];
  const before = JSON.stringify(layers);
  assert.equal(fitViewerZoom(1200, 700, layers), fitViewerZoom(1200, 700, [...layers].reverse()));
  assert.equal(viewerSceneBounds(layers).right, viewerSceneBounds([...layers].reverse()).right);
  assert.equal(viewerZoomAtPoint(1, { x: 0, y: 0 }, { x: 0, y: 0 }, 1).zoom, 1);
  assert.equal(JSON.stringify(layers), before);
});

test("zoom keeps the cursor anchor", () => {
  const result = viewerZoomAtPoint(1, { x: 20, y: -10 }, { x: 200, y: 120 }, 2);
  assert.deepEqual(result.pan, { x: -160, y: -140 });
});

test("resize handle inverse total scale keeps a 13px visual and 20px hit target fixed", () => {
  for (const [zoom, layerScale] of [[.05, .1], [.1, 1], [1, 1], [32, 8]]) {
    const inverse = inverseViewerHandleScale(zoom, layerScale);
    assert.ok(Math.abs(13 * zoom * layerScale * inverse - 13) < 1e-10);
    assert.ok(Math.abs(20 * zoom * layerScale * inverse - 20) < 1e-10);
  }
  assert.ok(Math.abs(inverseViewerHandleScale(.05, .1) - 200) < 1e-10);
  for (const result of [
    inverseViewerHandleScale(NaN, Infinity), inverseViewerHandleScale(-1, 0, 0), inverseViewerHandleScale(Number.MAX_VALUE, Number.MAX_VALUE),
    inverseViewerHandleScale(Number.MIN_VALUE, 1), inverseViewerHandleScale(5e-324, 1), inverseViewerHandleScale(1, -Infinity),
    inverseViewerHandleScale(1, 1, -1), inverseViewerHandleScale(1e300, 1e300), inverseViewerHandleScale(1e300, 1),
  ]) {
    assert.ok(Number.isFinite(result));
    assert.ok(result >= VIEWER_MIN_INVERSE_SCALE && result <= VIEWER_MAX_INVERSE_SCALE);
  }
  assert.equal(inverseViewerHandleScale(Number.MIN_VALUE, 1), 200);
  assert.equal(inverseViewerHandleScale(1e300, 1), VIEWER_MIN_INVERSE_SCALE);
  assert.ok(Math.abs(inverseViewerHandleScale(.05, .1, 1e300) - 200) < 1e-10);
});

test("handle metric helper preserves 13px visual and 20px hit target across all anchors' transforms", () => {
  for (const [zoom, scale] of [[.05, .1], [.1, 1], [1, 1], [32, 8]]) {
    const metrics = viewerHandleScreenMetrics(zoom, scale);
    assert.ok(Math.abs(metrics.visualPx - VIEWER_HANDLE_VISUAL_PX) < 1e-10);
    assert.ok(Math.abs(metrics.hitPx - VIEWER_HANDLE_HIT_PX) < 1e-10);
  }
});

test("resize change helper ignores zero and floating noise but retains scale and anchored position changes", () => {
  const initial = { scale: 1, x: 20, y: -10 };
  assert.equal(viewerResizeChanged(initial, initial), false);
  assert.equal(viewerResizeChanged(initial, { scale: 1 + 5e-7, x: 20, y: -10 }), false);
  assert.equal(viewerResizeChanged(initial, { scale: 1.01, x: 20, y: -10 }), true);
  assert.equal(viewerResizeChanged(initial, { scale: 1, x: 20.1, y: -10 }), true);
  assert.equal(viewerResizeChanged(initial, { scale: 1, x: 20, y: -10.1 }), true);
});

test("resize gesture final-transform admission permits return-to-origin painting without an undo", () => {
  const initial = { scale: 1, x: 20, y: -10 };
  const start = { active: true, initialTransform: initial, currentTransform: initial, changed: false, pointerId: 7 };
  const moved = viewerResizeGestureMove(start, { scale: 1.2, x: 10, y: -5 });
  assert.equal(moved.shouldApply, true);
  assert.equal(moved.changed, true);
  const returned = viewerResizeGestureMove(moved, initial);
  assert.equal(returned.shouldApply, true);
  assert.deepEqual(returned.currentTransform, initial);
  assert.equal(returned.changed, false);
  assert.equal(viewerResizeGestureFinish(returned).shouldSaveUndo, false);
});

test("resize gesture records one final change, while disable after return releases capture without undo", () => {
  const initial = { scale: 1, x: 0, y: 0 };
  const start = { active: true, initialTransform: initial, currentTransform: initial, changed: false, pointerId: 9 };
  const moved = viewerResizeGestureMove(start, { scale: 1.1, x: 4, y: 0 });
  assert.equal(viewerResizeGestureFinish(moved).shouldSaveUndo, true);
  const returned = viewerResizeGestureMove(moved, initial);
  const disabled = viewerResizeGestureFinish(returned, { disabled: true });
  assert.equal(disabled.shouldReleasePointer, true);
  assert.equal(disabled.shouldSaveUndo, false);
  assert.equal(viewerResizeGestureFinish(disabled.state).shouldSaveUndo, false);
});

test("resize disable decision is fail-closed and records one changed gesture", () => {
  assert.deepEqual(viewerResizeDisableDecision(null), { shouldFinish: false, shouldSaveUndo: false, shouldReleasePointer: false });
  assert.deepEqual(viewerResizeDisableDecision({ kind: "layer", changed: true, pointerId: 4 }), { shouldFinish: false, shouldSaveUndo: false, shouldReleasePointer: false });
  assert.deepEqual(viewerResizeDisableDecision({ kind: "resize", changed: false, pointerId: 4 }), { shouldFinish: true, shouldSaveUndo: false, shouldReleasePointer: true });
  assert.deepEqual(viewerResizeDisableDecision({ kind: "resize", changed: true, pointerId: 0 }), { shouldFinish: true, shouldSaveUndo: true, shouldReleasePointer: true });
});
