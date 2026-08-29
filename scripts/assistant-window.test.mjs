import assert from "node:assert/strict";
import test from "node:test";
import {
  MINIMUM_WINDOW_HEIGHT,
  MINIMUM_WINDOW_WIDTH,
  RESIZE_HANDLES,
  clampWindowRect,
  defaultWindowRect,
  isMaximizedRect,
  maximizedWindowRect,
  moveWindow,
  normalizeWindowRect,
  readStoredWindowRect,
  resizeWindow,
  writeStoredWindowRect,
} from "../src/assistant-window.js";

const viewport = { width: 1600, height: 900 };
const rect = Object.freeze({ x: 400, y: 200, width: 500, height: 600 });

const memoryStorage = () => {
  const store = new Map();
  return {
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => { store.set(key, String(value)); },
  };
};

test("the default window opens fully on screen against the right edge", () => {
  const opened = defaultWindowRect(viewport);
  assert.ok(opened.x + opened.width <= viewport.width);
  assert.ok(opened.y + opened.height <= viewport.height);
  assert.ok(opened.x > viewport.width / 2, "opens beside the composer, not over it");
  assert.ok(opened.width >= MINIMUM_WINDOW_WIDTH && opened.height >= MINIMUM_WINDOW_HEIGHT);
});

test("maximising fills the viewport but stays a floating panel, and is recognised as maximised", () => {
  const filled = maximizedWindowRect(viewport);
  assert.ok(filled.x > 0 && filled.y > 0, "the window keeps its margin: it never becomes a takeover layer");
  assert.ok(filled.x + filled.width < viewport.width);
  assert.ok(filled.y + filled.height < viewport.height);
  assert.ok(isMaximizedRect(filled, viewport));
  // The title bar offers maximise or restore based on the rect alone, so a reload that brings back a
  // maximised window still offers to restore it rather than to maximise it again.
  assert.ok(!isMaximizedRect(rect, viewport));
  assert.ok(!isMaximizedRect({ ...filled, width: filled.width - 40 }, viewport));
  // A viewport smaller than the minimum still yields a usable rect rather than a negative one.
  const tiny = maximizedWindowRect({ width: 200, height: 150 });
  assert.equal(tiny.width, MINIMUM_WINDOW_WIDTH);
  assert.equal(tiny.height, MINIMUM_WINDOW_HEIGHT);
  assert.ok(isMaximizedRect(tiny, { width: 200, height: 150 }));
});

test("dragging moves by the pointer delta and stops at the viewport edge", () => {
  assert.deepEqual(moveWindow(rect, 60, -40, viewport), { x: 460, y: 160, width: 500, height: 600 });
  // A window dragged past the edge must stay reachable; it can never be lost off-screen.
  const pushed = moveWindow(rect, 9000, 9000, viewport);
  assert.ok(pushed.x + pushed.width <= viewport.width);
  assert.ok(pushed.y + pushed.height <= viewport.height);
  const pulled = moveWindow(rect, -9000, -9000, viewport);
  assert.ok(pulled.x >= 0 && pulled.y >= 0);
});

test("every resize handle changes the size it should and pins the opposite edge", () => {
  assert.equal(resizeWindow(rect, "e", 100, 0, viewport).width, 600);
  assert.equal(resizeWindow(rect, "s", 0, 100, viewport).height, 700);

  // Dragging the west edge right must shrink the window without moving its right edge.
  const west = resizeWindow(rect, "w", 100, 0, viewport);
  assert.equal(west.width, 400);
  assert.equal(west.x + west.width, rect.x + rect.width);

  const north = resizeWindow(rect, "n", 100, 100, viewport);
  assert.equal(north.height, 500);
  assert.equal(north.y + north.height, rect.y + rect.height);

  const corner = resizeWindow(rect, "se", 80, 90, viewport);
  assert.deepEqual([corner.width, corner.height], [580, 690]);
  assert.deepEqual([corner.x, corner.y], [rect.x, rect.y]);
});

test("shrinking past the minimum stops without dragging the anchored edge along", () => {
  const west = resizeWindow(rect, "w", 9000, 0, viewport);
  assert.equal(west.width, MINIMUM_WINDOW_WIDTH);
  assert.equal(west.x + west.width, rect.x + rect.width, "the right edge must not creep");

  const north = resizeWindow(rect, "n", 0, 9000, viewport);
  assert.equal(north.height, MINIMUM_WINDOW_HEIGHT);
  assert.equal(north.y + north.height, rect.y + rect.height, "the bottom edge must not creep");
});

test("resizes are computed from the gesture's starting rect, so they never drift", () => {
  // Two events in one gesture: the second delta is measured from the same origin as the first.
  const first = resizeWindow(rect, "e", 50, 0, viewport);
  const second = resizeWindow(rect, "e", 100, 0, viewport);
  assert.equal(first.width, 550);
  assert.equal(second.width, 600, "an absolute delta must not compound");
  assert.deepEqual(resizeWindow(rect, "not-a-handle", 100, 100, viewport), clampWindowRect(rect, viewport));
  assert.deepEqual(RESIZE_HANDLES.slice().sort(), ["e", "n", "ne", "nw", "s", "se", "sw", "w"]);
});

test("a viewport smaller than the minimum still yields a usable rect", () => {
  const tiny = clampWindowRect(rect, { width: 200, height: 200 });
  assert.equal(tiny.width, MINIMUM_WINDOW_WIDTH);
  assert.equal(tiny.height, MINIMUM_WINDOW_HEIGHT);
  assert.ok(Number.isFinite(tiny.x) && Number.isFinite(tiny.y));
  assert.doesNotThrow(() => defaultWindowRect({ width: 0, height: 0 }));
});

test("a stored rect is restored, and a corrupt one falls back to the default", () => {
  const storage = memoryStorage();
  assert.deepEqual(readStoredWindowRect(viewport, storage), defaultWindowRect(viewport));
  writeStoredWindowRect(rect, storage);
  assert.deepEqual(readStoredWindowRect(viewport, storage), rect);

  assert.deepEqual(normalizeWindowRect({ x: 1, y: 2 }, viewport), defaultWindowRect(viewport));
  assert.deepEqual(normalizeWindowRect(null, viewport), defaultWindowRect(viewport));
  assert.deepEqual(normalizeWindowRect({ x: NaN, y: 0, width: 500, height: 600 }, viewport), defaultWindowRect(viewport));

  // A rect saved on a large monitor must be pulled back on screen on a smaller one.
  const smaller = { width: 800, height: 600 };
  const restored = normalizeWindowRect({ x: 1400, y: 700, width: 500, height: 600 }, smaller);
  assert.ok(restored.x + restored.width <= smaller.width);
  assert.ok(restored.y + restored.height <= smaller.height);
});

test("storage failures never break a drag", () => {
  assert.doesNotThrow(() => writeStoredWindowRect(rect, { setItem() { throw new Error("quota"); } }));
  assert.deepEqual(readStoredWindowRect(viewport, { getItem() { throw new Error("blocked"); } }), defaultWindowRect(viewport));
  assert.deepEqual(readStoredWindowRect(viewport, null), defaultWindowRect(viewport));
});
