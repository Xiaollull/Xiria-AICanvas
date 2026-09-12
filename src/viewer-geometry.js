export const VIEWER_MIN_ZOOM = 0.05;
export const VIEWER_MAX_ZOOM = 32;
export const VIEWER_HANDLE_VISUAL_PX = 13;
export const VIEWER_HANDLE_HIT_PX = 20;
export const VIEWER_MIN_TOTAL_SCALE = 0.005;
export const VIEWER_MAX_HANDLE_EPSILON = VIEWER_MIN_TOTAL_SCALE;
export const VIEWER_MAX_INVERSE_SCALE = 256;
export const VIEWER_MIN_INVERSE_SCALE = 1 / VIEWER_MAX_INVERSE_SCALE;
export const VIEWER_RESIZE_CHANGE_EPSILON = 0.000001;
export const VIEWER_BRUSH_CURSOR_MIN_PX = 3;
export const VIEWER_BRUSH_CURSOR_MAX_PX = 8192;

export function inverseViewerHandleScale(viewerZoom, layerScale, epsilon = 0.0001) {
  const zoom = Number(viewerZoom);
  const scale = Number(layerScale);
  const requestedFloor = Number(epsilon);
  const floor = Number.isFinite(requestedFloor) && requestedFloor > 0
    ? Math.max(VIEWER_MIN_TOTAL_SCALE, Math.min(VIEWER_MAX_HANDLE_EPSILON, requestedFloor))
    : VIEWER_MIN_TOTAL_SCALE;
  const product = Number.isFinite(zoom) && zoom > 0 && Number.isFinite(scale) && scale > 0 ? zoom * scale : floor;
  const denominator = Number.isFinite(product) && product > 0
    ? Math.max(floor, Math.min(1 / VIEWER_MIN_INVERSE_SCALE, product))
    : floor;
  const inverse = 1 / denominator;
  return Number.isFinite(inverse)
    ? Math.max(VIEWER_MIN_INVERSE_SCALE, Math.min(VIEWER_MAX_INVERSE_SCALE, inverse))
    : VIEWER_MAX_INVERSE_SCALE;
}

export function viewerHandleScreenMetrics(viewerZoom, layerScale) {
  const inverse = inverseViewerHandleScale(viewerZoom, layerScale);
  const total = Number(viewerZoom) * Number(layerScale);
  const screenScale = Number.isFinite(total) && total > 0 ? total * inverse : 1;
  return {
    inverse,
    visualPx: VIEWER_HANDLE_VISUAL_PX * screenScale,
    hitPx: VIEWER_HANDLE_HIT_PX * screenScale,
  };
}

// The brush ring is drawn at the size of the pixels the stroke will cover: a stroke is stored in
// the layer's own pixels, so the ring is brush pixels x that layer's scale x the viewer's zoom.
// It stays a DOM circle rather than a `cursor:` image because CSS cursors stop at 128px, while a
// 300px brush at 8x zoom needs far more; the floor keeps a one-pixel brush visible when zoomed out.
export function viewerBrushCursorSize(brushSize, viewerZoom, layerScale = 1) {
  const size = Number(brushSize);
  const zoom = Number(viewerZoom);
  const scale = Number(layerScale);
  const product = size * (Number.isFinite(zoom) && zoom > 0 ? zoom : 1) * (Number.isFinite(scale) && scale > 0 ? scale : 1);
  if (!Number.isFinite(product) || product <= 0) return VIEWER_BRUSH_CURSOR_MIN_PX;
  return Math.max(VIEWER_BRUSH_CURSOR_MIN_PX, Math.min(VIEWER_BRUSH_CURSOR_MAX_PX, product));
}

// The rectangle a drag has swept, normalised so dragging up or left describes the same box as
// dragging down or right.
export function viewerMarqueeRect(origin, current) {
  const coordinate = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
  const left = Math.min(coordinate(origin?.x), coordinate(current?.x));
  const top = Math.min(coordinate(origin?.y), coordinate(current?.y));
  const width = Math.abs(coordinate(current?.x) - coordinate(origin?.x));
  const height = Math.abs(coordinate(current?.y) - coordinate(origin?.y));
  return { x: left, y: top, width, height, centerX: left + width / 2, centerY: top + height / 2 };
}

export function viewerResizeChanged(initial, next, epsilon = VIEWER_RESIZE_CHANGE_EPSILON) {
  const requestedEpsilon = Number(epsilon);
  const threshold = Number.isFinite(requestedEpsilon) && requestedEpsilon > 0 ? requestedEpsilon : VIEWER_RESIZE_CHANGE_EPSILON;
  return ["scale", "x", "y"].some((key) => {
    const before = Number(initial?.[key]);
    const after = Number(next?.[key]);
    return Number.isFinite(before) && Number.isFinite(after) ? Math.abs(after - before) > threshold : !Object.is(before, after);
  });
}

// A resize gesture has two independent facts: whether its latest transform must
// be painted, and whether its final transform warrants an undo snapshot.
export function viewerResizeGestureMove(state, nextTransform) {
  if (!state?.active) return { ...state, shouldApply: false };
  const current = state.currentTransform || state.initialTransform;
  const shouldApply = viewerResizeChanged(current, nextTransform);
  return {
    ...state,
    currentTransform: shouldApply ? { ...nextTransform } : current,
    changed: viewerResizeChanged(state.initialTransform, nextTransform),
    shouldApply,
  };
}

export function viewerResizeGestureFinish(state, { disabled = false } = {}) {
  if (!state?.active) return { state: { ...state, shouldApply: false }, shouldSaveUndo: false, shouldReleasePointer: false };
  return {
    state: { ...state, active: false, shouldApply: false },
    shouldSaveUndo: state.changed === true,
    shouldReleasePointer: disabled && Number.isInteger(state.pointerId),
  };
}

// This is deliberately data-only so a resize-disable transition can be decided
// before React state is changed or a DOM pointer capture is released.
export function viewerResizeDisableDecision(drag) {
  const isResize = drag?.kind === "resize";
  return {
    shouldFinish: isResize,
    shouldSaveUndo: isResize && drag.changed === true,
    shouldReleasePointer: isResize && Number.isInteger(drag.pointerId),
  };
}

export function intrinsicDimensions(width, height) {
  const naturalWidth = Math.max(1, Math.round(Number(width) || 0));
  const naturalHeight = Math.max(1, Math.round(Number(height) || 0));
  return { naturalWidth, naturalHeight };
}

export function viewerLayerBounds(layer) {
  return viewerEditorLayerBounds(layer);
}

export function viewerSceneBounds(layers) {
  const bounds = layers.map(viewerLayerBounds);
  if (!bounds.length) return null;
  return {
    left: Math.min(...bounds.map((item) => item.left)),
    right: Math.max(...bounds.map((item) => item.right)),
    top: Math.min(...bounds.map((item) => item.top)),
    bottom: Math.max(...bounds.map((item) => item.bottom)),
  };
}

export function fitViewerZoom(viewportWidth, viewportHeight, layers, padding = 48) {
  const bounds = viewerSceneBounds(layers);
  if (!bounds) return 1;
  const availableWidth = Math.max(1, viewportWidth - padding * 2);
  const availableHeight = Math.max(1, viewportHeight - padding * 2);
  return Math.max(VIEWER_MIN_ZOOM, Math.min(1, availableWidth / (bounds.right - bounds.left), availableHeight / (bounds.bottom - bounds.top)));
}

export function viewerZoomAtPoint(currentZoom, currentPan, pointer, factor) {
  const nextZoom = Math.max(VIEWER_MIN_ZOOM, Math.min(VIEWER_MAX_ZOOM, currentZoom * factor));
  const ratio = nextZoom / currentZoom;
  return { zoom: nextZoom, pan: { x: pointer.x - (pointer.x - currentPan.x) * ratio, y: pointer.y - (pointer.y - currentPan.y) * ratio } };
}
import { viewerEditorLayerBounds } from "./viewer-editor.js";
