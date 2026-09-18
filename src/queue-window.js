/**
 * Where the queue window sits, and how it stays on screen.
 *
 * It used to be an absolutely positioned child of the button that opens it, which put it inside the
 * result panel's scrollport -- so the panel clipped it, and the left edge of the window (its title,
 * and the thumbnail of every task in it) was simply cut off. A floating window belongs to the
 * viewport, not to the column that happens to contain its trigger, so it is portalled to the
 * document and positioned in viewport coordinates.
 *
 * Which then makes it draggable, because the only thing standing between "fixed under the button"
 * and "wherever the user wants it" is where these two numbers come from. The geometry lives here,
 * with no DOM in it, so the rules can be read and tested directly.
 */

// How close to the edge of the window it may be pushed. Enough that the drop shadow still reads as
// a window rather than as something falling off the screen.
export const QUEUE_WINDOW_MARGIN = 12;
// The gap between the button and the window that opens under it.
export const QUEUE_WINDOW_ANCHOR_GAP = 8;
// What the stylesheet gives it, used to place the window before it has been measured.
export const QUEUE_WINDOW_DEFAULT_SIZE = Object.freeze({ width: 408, height: 610 });

function clampAxis(value, extent, viewportExtent) {
  // When the window is larger than the viewport it is pinned to the top/left edge instead of being
  // pushed off the other one: the header is what the user needs to reach, and it is at that end.
  const limit = viewportExtent - extent - QUEUE_WINDOW_MARGIN;
  if (limit <= QUEUE_WINDOW_MARGIN) return QUEUE_WINDOW_MARGIN;
  return Math.min(limit, Math.max(QUEUE_WINDOW_MARGIN, value));
}

/** Keep a position fully on screen. Applied on open, on drag, and after the viewport resizes. */
export function clampQueueWindow(position, size, viewport) {
  const width = Number(size?.width) || QUEUE_WINDOW_DEFAULT_SIZE.width;
  const height = Number(size?.height) || QUEUE_WINDOW_DEFAULT_SIZE.height;
  const viewWidth = Number(viewport?.width) || 0;
  const viewHeight = Number(viewport?.height) || 0;
  return {
    x: clampAxis(Number(position?.x) || 0, width, viewWidth),
    y: clampAxis(Number(position?.y) || 0, height, viewHeight),
  };
}

/**
 * Where the window opens the first time: under the button that opened it, right edges aligned, so
 * it reads as belonging to that button before the user has moved it anywhere.
 */
export function queueWindowPlacement(anchorRect, size, viewport) {
  const width = Number(size?.width) || QUEUE_WINDOW_DEFAULT_SIZE.width;
  const height = Number(size?.height) || QUEUE_WINDOW_DEFAULT_SIZE.height;
  if (!anchorRect) {
    return clampQueueWindow({ x: (Number(viewport?.width) || 0) - width - QUEUE_WINDOW_MARGIN, y: QUEUE_WINDOW_MARGIN }, size, viewport);
  }
  const below = Number(anchorRect.bottom) + QUEUE_WINDOW_ANCHOR_GAP;
  const viewHeight = Number(viewport?.height) || 0;
  // If it will not fit below the button, open it above rather than letting the clamp slide it back
  // over the button it came from.
  const fitsBelow = below + height + QUEUE_WINDOW_MARGIN <= viewHeight;
  const above = Number(anchorRect.top) - QUEUE_WINDOW_ANCHOR_GAP - height;
  return clampQueueWindow({ x: Number(anchorRect.right) - width, y: fitsBelow ? below : above }, size, viewport);
}

/**
 * The position a drag has reached.
 *
 * `origin` is where the window was when the gesture began together with the pointer that began it,
 * so the window moves with the pointer rather than jumping its own corner under it.
 */
export function queueWindowDrag(origin, pointer, size, viewport) {
  return clampQueueWindow({
    x: Number(origin?.x) + (Number(pointer?.x) - Number(origin?.pointerX)),
    y: Number(origin?.y) + (Number(pointer?.y) - Number(origin?.pointerY)),
  }, size, viewport);
}
