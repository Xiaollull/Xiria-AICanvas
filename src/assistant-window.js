// Geometry for the floating assistant window.
//
// Split out from the component so drag and resize arithmetic is unit-tested rather than verified by
// dragging things with a mouse. Every function is pure: it takes a rect plus the pointer delta from
// the gesture's starting rect and returns a new rect, so a gesture never accumulates rounding drift.

export const MINIMUM_WINDOW_WIDTH = 360;
export const MINIMUM_WINDOW_HEIGHT = 380;
export const WINDOW_STORAGE_KEY = "xirai-assistant-window-v1";

// Leaves the window fully on screen. A floating panel that can be dragged under the viewport edge
// is unrecoverable without clearing storage, and the pop-out button is the intended way to get a
// bigger surface anyway.
const EDGE_MARGIN = 8;

export const RESIZE_HANDLES = Object.freeze(["n", "s", "e", "w", "ne", "nw", "se", "sw"]);

const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));

function bounds(viewport) {
  const width = Number.isFinite(viewport?.width) ? viewport.width : 0;
  const height = Number.isFinite(viewport?.height) ? viewport.height : 0;
  return {
    width,
    height,
    // A viewport narrower than the minimum still has to produce a usable rect, so the maximum is
    // allowed to fall below the minimum and the clamp order below resolves it.
    maximumWidth: Math.max(MINIMUM_WINDOW_WIDTH, width - EDGE_MARGIN * 2),
    maximumHeight: Math.max(MINIMUM_WINDOW_HEIGHT, height - EDGE_MARGIN * 2),
  };
}

export function clampWindowRect(rect, viewport) {
  const { width: viewportWidth, height: viewportHeight, maximumWidth, maximumHeight } = bounds(viewport);
  const width = clamp(Math.round(rect.width), MINIMUM_WINDOW_WIDTH, maximumWidth);
  const height = clamp(Math.round(rect.height), MINIMUM_WINDOW_HEIGHT, maximumHeight);
  const maximumX = Math.max(EDGE_MARGIN, viewportWidth - width - EDGE_MARGIN);
  const maximumY = Math.max(EDGE_MARGIN, viewportHeight - height - EDGE_MARGIN);
  return {
    x: clamp(Math.round(rect.x), EDGE_MARGIN, maximumX),
    y: clamp(Math.round(rect.y), EDGE_MARGIN, maximumY),
    width,
    height,
  };
}

export function defaultWindowRect(viewport) {
  const { width, height } = bounds(viewport);
  const windowWidth = clamp(Math.round(width * 0.34), MINIMUM_WINDOW_WIDTH, 560);
  const windowHeight = clamp(Math.round(height * 0.68), MINIMUM_WINDOW_HEIGHT, 760);
  // Opens against the right edge: the prompt composer it serves sits in the centre column, and a
  // centred window would cover the text the user is about to rewrite.
  return clampWindowRect({
    x: width - windowWidth - 32,
    y: Math.round((height - windowHeight) / 2),
    width: windowWidth,
    height: windowHeight,
  }, viewport);
}

// Fills the viewport but keeps the same margin every other rect respects, so a maximised window is
// still visibly a floating panel over the workspace rather than a takeover of it.
export function maximizedWindowRect(viewport) {
  const { width, height } = bounds(viewport);
  return clampWindowRect({ x: EDGE_MARGIN, y: EDGE_MARGIN, width: width - EDGE_MARGIN * 2, height: height - EDGE_MARGIN * 2 }, viewport);
}

// Derived from geometry rather than tracked as a flag: the rect survives a reload and the flag would
// not, which would leave the title bar offering to maximise an already maximised window.
export function isMaximizedRect(rect, viewport) {
  const filled = maximizedWindowRect(viewport);
  return Math.round(rect.width) >= filled.width && Math.round(rect.height) >= filled.height;
}

export function moveWindow(startRect, deltaX, deltaY, viewport) {
  return clampWindowRect({ ...startRect, x: startRect.x + deltaX, y: startRect.y + deltaY }, viewport);
}

// `startRect` is the rect when the gesture began, not the live one, so the pointer stays glued to
// the handle it grabbed even after a clamp truncates one axis.
export function resizeWindow(startRect, handle, deltaX, deltaY, viewport) {
  if (!RESIZE_HANDLES.includes(handle)) return clampWindowRect(startRect, viewport);
  const { maximumWidth, maximumHeight } = bounds(viewport);
  const right = startRect.x + startRect.width;
  const bottom = startRect.y + startRect.height;
  let { x, y, width, height } = startRect;
  if (handle.includes("e")) width = clamp(startRect.width + deltaX, MINIMUM_WINDOW_WIDTH, maximumWidth);
  if (handle.includes("s")) height = clamp(startRect.height + deltaY, MINIMUM_WINDOW_HEIGHT, maximumHeight);
  // Dragging a west or north edge pins the opposite edge: the anchored side must not creep when the
  // size hits its minimum.
  if (handle.includes("w")) {
    width = clamp(startRect.width - deltaX, MINIMUM_WINDOW_WIDTH, maximumWidth);
    x = right - width;
  }
  if (handle.includes("n")) {
    height = clamp(startRect.height - deltaY, MINIMUM_WINDOW_HEIGHT, maximumHeight);
    y = bottom - height;
  }
  return clampWindowRect({ x, y, width, height }, viewport);
}

export function normalizeWindowRect(value, viewport) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : null;
  if (!source || !["x", "y", "width", "height"].every((key) => Number.isFinite(source[key]))) {
    return defaultWindowRect(viewport);
  }
  return clampWindowRect(source, viewport);
}

export function readStoredWindowRect(viewport, storage) {
  try {
    const saved = storage?.getItem(WINDOW_STORAGE_KEY);
    return normalizeWindowRect(saved ? JSON.parse(saved) : null, viewport);
  } catch {
    return defaultWindowRect(viewport);
  }
}

export function writeStoredWindowRect(rect, storage) {
  try {
    storage?.setItem(WINDOW_STORAGE_KEY, JSON.stringify(rect));
  } catch {
    // A full or blocked storage quota must not break dragging.
  }
}
