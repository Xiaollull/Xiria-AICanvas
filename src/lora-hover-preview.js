// Placement for the large preview that follows a hovered LoRA row.
//
// The rows it anchors to sit in a narrow left rail, in a modal, and on a full
// page, so the panel cannot assume a side: it takes whichever one it fits in,
// prefers the right when both work, and is clamped so it is never partly off
// screen. All of that is arithmetic, so it lives here where a test can check the
// awkward cases without a browser.

export const HOVER_PREVIEW_WIDTH = 300;
export const HOVER_PREVIEW_GAP = 14;
export const HOVER_PREVIEW_MARGIN = 12;
/** Long enough that sliding the pointer down a mounted list does not strobe. */
export const HOVER_PREVIEW_DELAY = 200;

function clamp(value, lowest, highest) {
  // A viewport shorter than the panel makes the range empty; pinning to the top
  // edge at least keeps the head of the card visible.
  if (highest < lowest) return lowest;
  return Math.min(highest, Math.max(lowest, value));
}

export function hoverPreviewPlacement({
  anchor,
  viewport,
  size,
  gap = HOVER_PREVIEW_GAP,
  margin = HOVER_PREVIEW_MARGIN,
} = {}) {
  const anchorRect = {
    left: Number(anchor?.left) || 0,
    top: Number(anchor?.top) || 0,
    right: Number(anchor?.right) || 0,
    bottom: Number(anchor?.bottom) || 0,
  };
  const width = Math.max(0, Number(size?.width) || 0);
  const height = Math.max(0, Number(size?.height) || 0);
  const viewWidth = Math.max(0, Number(viewport?.width) || 0);
  const viewHeight = Math.max(0, Number(viewport?.height) || 0);

  const roomRight = viewWidth - anchorRect.right - gap - margin;
  const roomLeft = anchorRect.left - gap - margin;
  if (roomRight >= width || roomLeft >= width) {
    const side = roomRight >= width ? "right" : "left";
    const left = clamp(side === "right" ? anchorRect.right + gap : anchorRect.left - gap - width, margin, viewWidth - width - margin);
    // Centred on the row rather than pinned to its top edge: a 44px row beside a
    // 400px panel otherwise pushes the whole picture below the pointer.
    const centred = anchorRect.top + (anchorRect.bottom - anchorRect.top) / 2 - height / 2;
    return { left: Math.round(left), top: Math.round(clamp(centred, margin, viewHeight - height - margin)), side };
  }

  // Neither side has room — a narrow window, or a row in a wide panel. Squeezing
  // it in beside the row anyway would put the picture on top of the thing the
  // pointer is resting on, so it goes under the row instead, or over it when
  // there is more space that way.
  const roomBelow = viewHeight - anchorRect.bottom - gap - margin;
  const roomAbove = anchorRect.top - gap - margin;
  const side = roomBelow >= height || roomBelow >= roomAbove ? "below" : "above";
  const top = side === "below" ? anchorRect.bottom + gap : anchorRect.top - gap - height;
  return {
    left: Math.round(clamp(anchorRect.left, margin, viewWidth - width - margin)),
    top: Math.round(clamp(top, margin, viewHeight - height - margin)),
    side,
  };
}

/**
 * Whether a row can open a preview at all.
 *
 * Deliberately satisfied by a bare file name. It used to require a cover, a
 * prompt, a note or a tag, so hovering a LoRA nobody had customised yet did
 * nothing — indistinguishable from the feature being broken, and with no hint
 * that a card was there to be filled in. The panel now opens for any real row
 * and says what is missing.
 */
export function hasHoverPreview(presentation) {
  return Boolean(presentation?.title || presentation?.fileName || presentation?.coverUrl);
}
