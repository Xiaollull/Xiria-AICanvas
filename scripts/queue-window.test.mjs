import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  QUEUE_WINDOW_ANCHOR_GAP,
  QUEUE_WINDOW_DEFAULT_SIZE,
  QUEUE_WINDOW_MARGIN,
  clampQueueWindow,
  queueWindowDrag,
  queueWindowPlacement,
} from "../src/queue-window.js";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
const size = { width: 408, height: 610 };
const viewport = { width: 1600, height: 900 };

test("the window opens under the button that opened it", () => {
  const anchor = { top: 60, bottom: 92, left: 1200, right: 1320 };
  const placed = queueWindowPlacement(anchor, size, viewport);
  // Right edges aligned, so it reads as belonging to that button before it has been moved.
  assert.equal(placed.x, anchor.right - size.width);
  assert.equal(placed.y, anchor.bottom + QUEUE_WINDOW_ANCHOR_GAP);
});

test("a window that will not fit below the button opens above it", () => {
  // Otherwise the clamp slides it back up over the button it came from, which hides the thing the
  // user just pressed behind the thing it opened.
  const low = { top: 700, bottom: 740, left: 1200, right: 1320 };
  const placed = queueWindowPlacement(low, size, viewport);
  assert.equal(placed.y, low.top - QUEUE_WINDOW_ANCHOR_GAP - size.height);
  assert.ok(placed.y + size.height < low.top, "it clears the button");
});

test("the window is always fully on screen", () => {
  // Dragged hard into each corner.
  assert.deepEqual(clampQueueWindow({ x: -900, y: -900 }, size, viewport), { x: QUEUE_WINDOW_MARGIN, y: QUEUE_WINDOW_MARGIN });
  assert.deepEqual(clampQueueWindow({ x: 9000, y: 9000 }, size, viewport), {
    x: viewport.width - size.width - QUEUE_WINDOW_MARGIN,
    y: viewport.height - size.height - QUEUE_WINDOW_MARGIN,
  });
  // A window taller than the viewport pins to the top edge rather than the bottom: the header is
  // what has to stay reachable, and the header is at that end.
  const short = { width: 600, height: 500 };
  assert.deepEqual(clampQueueWindow({ x: 0, y: 400 }, { width: 900, height: 900 }, short), { x: QUEUE_WINDOW_MARGIN, y: QUEUE_WINDOW_MARGIN });
  // Missing measurements fall back to the size the stylesheet gives it rather than collapsing to 0.
  assert.deepEqual(clampQueueWindow({ x: 9000, y: 0 }, null, viewport), {
    x: viewport.width - QUEUE_WINDOW_DEFAULT_SIZE.width - QUEUE_WINDOW_MARGIN,
    y: QUEUE_WINDOW_MARGIN,
  });
});

test("a drag moves the window with the pointer, not its corner under it", () => {
  const origin = { x: 300, y: 200, pointerX: 420, pointerY: 230 };
  // Grabbed 120px into the header and moved 60 right, 45 down.
  const moved = queueWindowDrag(origin, { x: 480, y: 275 }, size, viewport);
  assert.deepEqual(moved, { x: 360, y: 245 });
  // The same clamp applies mid-drag, so it cannot be thrown off the screen and lost.
  const thrown = queueWindowDrag(origin, { x: 99999, y: 99999 }, size, viewport);
  assert.deepEqual(thrown, { x: viewport.width - size.width - QUEUE_WINDOW_MARGIN, y: viewport.height - size.height - QUEUE_WINDOW_MARGIN });
});

test("the window is portalled out of the panel that was clipping it", async () => {
  const app = await read("src/App.jsx");
  const css = await read("src/styles.css");
  // It was an absolutely positioned child of its own button, inside the result panel's scrollport,
  // so that panel clipped its left edge -- the title and every task thumbnail with it.
  assert.match(app, /return createPortal\(/);
  assert.match(app, /<\/div>,\s*document\.body,\s*\);/);
  assert.match(css, /\.queue-dropdown \{ position: fixed;/);
  assert.doesNotMatch(css, /\.queue-dropdown \{ position: absolute/);
  // Its stacking number comes from the shared floating-window stack, so the queue and the preset
  // window come to the front in the order they were last used rather than by fixed values.
  assert.match(app, /style=\{\{ left: `\$\{position\?\.x \?\? 0\}px`, top: `\$\{position\?\.y \?\? 0\}px`, zIndex: layer\.zIndex \}\}/);
  assert.match(app, /const layer = useFloatingWindowLayer\("queue"\);/);
});

test("the window can be dragged anywhere, and stays where it was put", async () => {
  const app = await read("src/App.jsx");
  const css = await read("src/styles.css");
  assert.match(app, /onPointerDown=\{startDrag\}/);
  assert.match(app, /onPointerMove=\{moveDrag\}/);
  for (const ender of ["onPointerUp={endDrag}", "onPointerCancel={endDrag}"]) assert.ok(app.includes(ender), `${ender} releases the gesture`);
  assert.match(app, /event\.currentTarget\.setPointerCapture\?\.\(event\.pointerId\)/);
  assert.match(app, /if \(drag\.pointerId !== event\.pointerId\) return;|drag\.pointerId !== event\.pointerId\) return;/,
    "a second pointer must not steer someone else's drag");
  // The close button lives in the header, which is the drag handle.
  assert.match(app, /if \(event\.button !== 0 \|\| event\.target\.closest\("button"\)\) return;/);
  // A finger has to move the window rather than scroll the page under it.
  assert.match(css, /\.queue-dropdown > header \{ cursor: grab; touch-action: none; \}/);
  // Kept in the page's state, not the window's, so closing and reopening does not undo the move.
  assert.match(app, /const \[queueWindowPosition, setQueueWindowPosition\] = useState\(null\);/);
  assert.match(app, /if \(!current && !queueWindowPosition\)/, "only placed on the first open");
  assert.match(app, /position=\{queueWindowPosition\} onMove=\{setQueueWindowPosition\}/);
  // A window parked near an edge must not be left off-screen by a resize.
  assert.match(app, /window\.addEventListener\("resize", onResize\)/);
});

test("clicking a task no longer closes the window", async () => {
  const app = await read("src/App.jsx");
  const selectJob = app.slice(app.indexOf("  const selectQueueJob ="), app.indexOf("  // When the watched job ends"));
  assert.ok(selectJob.length > 0, "selectQueueJob is still there to check");
  assert.doesNotMatch(selectJob, /setQueueOpen\(false\)/,
    "selecting a task used to close the window out from under the user; only the close button does that now");
  // The close button is still the way out.
  assert.match(app, /onClose=\{\(\) => setQueueOpen\(false\)\}/);
  assert.match(app, /className="icon-button" title="关闭队列" onClick=\{onClose\}/);
});
