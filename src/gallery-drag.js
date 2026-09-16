/**
 * When a press on a gallery card becomes a sort, and when it stays a click.
 *
 * A mouse and a finger want opposite things from the same gesture. A mouse has a button, so
 * pressing one already says "I mean this card" and the drag can start the moment it moves --
 * waiting a third of a second first is the thing that made desktop sorting feel broken. A finger
 * has no button: pressing and moving is how the page is scrolled, so a touch has to be held still
 * for a moment before it can mean anything else, and it has to tolerate the wobble a finger has and
 * a mouse does not.
 *
 * Kept out of the component because it is a set of rules with no DOM in it, and because the numbers
 * are the whole behaviour: they belong somewhere they can be read and tested directly.
 */

export const GALLERY_DRAG_HOLD_MS = 320;
// A mouse pointer that has moved this far is being dragged, not clicked.
export const GALLERY_DRAG_MOUSE_THRESHOLD = 6;
// How far a finger may wander during the hold before it is treated as a scroll instead.
export const GALLERY_DRAG_TOUCH_TOLERANCE = 14;
// A click arriving after a touch can lag the release by roughly a quarter second; the swallow has
// to outlive that, but must not sit there forever if no click ever comes.
export const GALLERY_CLICK_SWALLOW_MS = 800;
// Edge band that auto-scrolls while dragging, and the fastest it may scroll per frame.
export const GALLERY_AUTOSCROLL_EDGE = 72;
export const GALLERY_AUTOSCROLL_MAX_STEP = 18;

export function galleryPointerIsCoarse(pointerType) {
  return pointerType === "touch" || pointerType === "pen";
}

/**
 * What a press should do, given how far it has moved and how long it has been held.
 *
 * - `drag`: start sorting now.
 * - `release`: this press is a scroll or a stray move; stop watching it and let the browser have it.
 * - `wait`: nothing yet.
 */
export function galleryPressIntent({ pointerType, distance, elapsed, held = false }) {
  const coarse = galleryPointerIsCoarse(pointerType);
  if (coarse) {
    // The hold is what separates "I want to move this" from "I want to scroll the page".
    if (!held) return distance > GALLERY_DRAG_TOUCH_TOLERANCE ? "release" : elapsed >= GALLERY_DRAG_HOLD_MS ? "drag" : "wait";
    return "drag";
  }
  if (held) return "drag";
  return distance > GALLERY_DRAG_MOUSE_THRESHOLD ? "drag" : "wait";
}

/**
 * How fast to scroll when the pointer is near an edge of the scrolling area.
 *
 * Proportional to how far into the band the pointer has reached, so the list creeps when the
 * pointer is at the edge of the band and runs when it is against the boundary. Returns 0 when the
 * pointer is clear of both bands, which is what stops the loop.
 */
export function galleryAutoScrollStep(pointerY, bounds, { edge = GALLERY_AUTOSCROLL_EDGE, max = GALLERY_AUTOSCROLL_MAX_STEP } = {}) {
  if (!bounds || !Number.isFinite(pointerY)) return 0;
  const depth = (distance) => Math.min(1, Math.max(0, (edge - distance) / edge));
  const above = depth(pointerY - bounds.top);
  const below = depth(bounds.bottom - pointerY);
  if (above > 0 && above >= below) return -Math.ceil(above * max);
  if (below > 0) return Math.ceil(below * max);
  return 0;
}

/**
 * Where a drag would land, and what happens on release.
 *
 * The list is frozen for the whole drag: nothing splices while the pointer is moving. That is not
 * only a matter of feel. The grid is dealt into columns round-robin, so a card whose index changes
 * usually changes column too, and React moves it by unmounting it from one column and mounting it
 * in the other. The node that held the pointer capture stops existing, the browser releases the
 * capture with it, and the gesture is orphaned mid-drag -- the card never arrives and the ghost
 * never goes away. Reordering on release means the dragged node is never rebuilt under the pointer.
 *
 * The drop point is an anchor rather than a number because the grid may be filtered. "Before this
 * card" survives a translation into the full collection order; the eighth visible position does not.
 */

export const GALLERY_DROP_HEAD = "head";
export const GALLERY_DROP_TAIL = "tail";

/**
 * The anchor a pointer is over, given the laid-out cards.
 *
 * `cards` are `{ id, rect }` in visual order. The nearest column is chosen by x, then the pointer's
 * y against each card's midpoint decides whether it lands before or after it -- the same rule a
 * text caret follows between two characters.
 */
export function galleryDropAnchor(cards, pointer) {
  const laid = (Array.isArray(cards) ? cards : []).filter((card) => card?.id && card.rect);
  if (!laid.length || !pointer) return { type: GALLERY_DROP_TAIL };
  const columnOf = (rect) => Math.round(rect.left);
  const columns = [...new Set(laid.map((card) => columnOf(card.rect)))].sort((first, second) => first - second);
  const nearest = columns.reduce((best, left) => {
    const width = laid.find((card) => columnOf(card.rect) === left)?.rect.width || 0;
    const distance = Math.abs(pointer.x - (left + width / 2));
    return !best || distance < best.distance ? { left, distance } : best;
  }, null);
  const column = laid.filter((card) => columnOf(card.rect) === nearest.left);
  for (const card of column) {
    if (pointer.y < card.rect.top + card.rect.height / 2) return { type: "before", cardId: card.id };
  }
  const last = column.at(-1);
  // Past the bottom of a column: after its last card, which in visual order is "before whatever
  // follows it", and the tail of the list when nothing does.
  const index = laid.findIndex((card) => card.id === last.id);
  const next = laid[index + 1];
  return next ? { type: "before", cardId: next.id } : { type: GALLERY_DROP_TAIL };
}

/**
 * The anchor as an index into the full collection order.
 *
 * Under a filter the anchor names a card the user can actually see, so the hidden cards around it
 * keep their places instead of being swept to one end.
 */
export function resolveDropIndex(fullOrder, anchor, movingId = null) {
  const order = (Array.isArray(fullOrder) ? fullOrder : []).filter((id) => id !== movingId);
  if (!anchor || anchor.type === GALLERY_DROP_TAIL) return order.length;
  if (anchor.type === GALLERY_DROP_HEAD) return 0;
  const index = order.indexOf(anchor.cardId);
  return index < 0 ? order.length : index;
}

/** Move one id to `index`, counted in the order that remains once it has been lifted out. */
export function spliceGalleryOrder(order, cardId, index) {
  const list = Array.isArray(order) ? [...order] : [];
  const from = list.indexOf(cardId);
  if (from < 0) return list;
  list.splice(from, 1);
  const target = Math.max(0, Math.min(list.length, Math.trunc(index)));
  list.splice(target, 0, cardId);
  return list;
}

/** Whether landing here would change anything. A no-op release must not send a request. */
export function galleryOrderChanged(before, after) {
  const first = Array.isArray(before) ? before : [];
  const second = Array.isArray(after) ? after : [];
  return first.length !== second.length || first.some((id, index) => id !== second[index]);
}

/**
 * Which visible cards yield to open the slot, and by how far.
 *
 * Only the target column moves: the cards at or after the insertion point slide down by one card,
 * which is what makes the empty slot appear where the card would land. Returned as a map so the
 * component can hand each tile a single number and let the stylesheet animate it.
 */
export function gallerySlotShifts(cards, anchor, movingId, columnCount) {
  const shifts = new Map();
  const laid = (Array.isArray(cards) ? cards : []).filter((card) => card?.id && card.rect);
  if (!laid.length || !anchor || anchor.type === GALLERY_DROP_TAIL) return shifts;
  const remaining = laid.filter((card) => card.id !== movingId);
  const anchorIndex = anchor.type === GALLERY_DROP_HEAD ? 0 : remaining.findIndex((card) => card.id === anchor.cardId);
  if (anchorIndex < 0) return shifts;
  const count = Math.max(1, Math.trunc(columnCount) || 1);
  const targetColumn = anchorIndex % count;
  const moving = laid.find((card) => card.id === movingId);
  const height = (moving?.rect.height || remaining[0]?.rect.height || 0);
  if (!height) return shifts;
  remaining.forEach((card, index) => {
    if (index >= anchorIndex && index % count === targetColumn) shifts.set(card.id, height);
  });
  return shifts;
}
