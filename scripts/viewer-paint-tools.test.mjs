import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { viewerBrushCursorSize, viewerMarqueeRect, VIEWER_BRUSH_CURSOR_MAX_PX, VIEWER_BRUSH_CURSOR_MIN_PX } from "../src/viewer-geometry.js";
import {
  drawTextLayer,
  normalizeManualLayout,
  normalizeTextLayer,
  serializeViewerLayer,
  viewerSafeResizeHandles,
  viewerTextBox,
  viewerTextBoxResize,
  wrapTextLines,
  VIEWER_TEXT_MIN_BOX,
  VIEWER_TEXT_PADDING,
} from "../src/viewer-editor.js";

const tenPixelsPerCharacter = (value) => value.length * 10;

function recordingContext() {
  const calls = [];
  return {
    calls,
    save() { calls.push(["save"]); },
    restore() { calls.push(["restore"]); },
    beginPath() { calls.push(["beginPath"]); },
    rect(...args) { calls.push(["rect", ...args]); },
    clip() { calls.push(["clip"]); },
    fillText(...args) { calls.push(["fillText", ...args]); },
    measureText(value) { return { width: tenPixelsPerCharacter(value) }; },
  };
}

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

test("a swept rectangle is the same box whichever corner the drag started from", () => {
  const forward = viewerMarqueeRect({ x: -10, y: -4 }, { x: 30, y: 16 });
  const backward = viewerMarqueeRect({ x: 30, y: 16 }, { x: -10, y: -4 });
  assert.deepEqual(forward, backward);
  assert.deepEqual(forward, { x: -10, y: -4, width: 40, height: 20, centerX: 10, centerY: 6 });
  assert.deepEqual(viewerMarqueeRect({ x: 5, y: 5 }, { x: 5, y: 5 }), { x: 5, y: 5, width: 0, height: 0, centerX: 5, centerY: 5 });
  assert.equal(viewerMarqueeRect(null, undefined).width, 0);
});

test("text wraps where the box makes it wrap, and only there", () => {
  const limit = 100;
  assert.deepEqual(wrapTextLines("hello world there", limit, tenPixelsPerCharacter), ["hello ", "world ", "there"]);
  // A word with nowhere to break still has to fit, the way `overflow-wrap: anywhere` breaks it.
  assert.deepEqual(wrapTextLines("supercalifragilistic", limit, tenPixelsPerCharacter), ["supercalif", "ragilistic"]);
  assert.deepEqual(wrapTextLines("one\ntwo", limit, tenPixelsPerCharacter), ["one", "two"]);
  assert.deepEqual(wrapTextLines("a\n\nb", limit, tenPixelsPerCharacter), ["a", "", "b"]);
  // Trailing spaces hang past the edge rather than pushing a line over it.
  assert.deepEqual(wrapTextLines("abcdefghij   ", limit, tenPixelsPerCharacter), ["abcdefghij   "]);
  // Without a width there is no wrapping: only the newlines the text itself carries.
  assert.deepEqual(wrapTextLines("one two three", 0, tenPixelsPerCharacter), ["one two three"]);
  // A box only ever shows a few lines, and wrapping the ones it hides costs work nobody sees.
  assert.deepEqual(wrapTextLines("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", limit, tenPixelsPerCharacter, 2), ["aaaaaaaaaa", "aaaaaaaaaa"]);
});

test("a small box wraps what it shows, not the text it clips", () => {
  // Two lines of 48px text fit in 140px; the rest is clipped, so the rest is never laid out.
  const cramped = normalizeTextLayer({ text: "a".repeat(4000), boxWidth: 64, boxHeight: 140 }, tenPixelsPerCharacter);
  assert.ok(cramped.lines.length <= 4, `wrapped ${cramped.lines.length} lines into a box that shows two`);
  assert.deepEqual([cramped.naturalWidth, cramped.naturalHeight], [64, 140]);
  // Clipping is a display decision: the layer still carries every character the user typed.
  assert.equal(cramped.text.length, 4000);
});

test("a dragged text box keeps its size, wraps inside it, and survives a layout round trip", () => {
  const dragged = normalizeTextLayer({ text: "hello world there", boxWidth: 124, boxHeight: 200, x: 4, y: 8 }, tenPixelsPerCharacter);
  assert.deepEqual(viewerTextBox(dragged), { width: 124, height: 200 });
  // The box is the layer: its size comes from the drag, not from how wide the glyphs happen to be.
  assert.deepEqual([dragged.naturalWidth, dragged.naturalHeight], [124, 200]);
  assert.deepEqual(dragged.lines, ["hello ", "world ", "there"]);
  const serialized = serializeViewerLayer(dragged);
  assert.deepEqual([serialized.boxWidth, serialized.boxHeight], [124, 200]);
  const restored = normalizeManualLayout({ version: 2, layers: [serialized] });
  assert.deepEqual([restored.layers[0].boxWidth, restored.layers[0].boxHeight], [124, 200]);
  // Text that grows with its own glyphs carries no box, and nothing about it changed.
  const grown = normalizeTextLayer({ text: "hello" }, tenPixelsPerCharacter);
  assert.equal(viewerTextBox(grown), null);
  assert.equal(serializeViewerLayer(grown).boxWidth, undefined);
  assert.equal(grown.naturalWidth, 50 + VIEWER_TEXT_PADDING * 2);
});

test("a box offers every edge; text that is only as wide as its glyphs offers corners", () => {
  const boxed = normalizeTextLayer({ text: "x", boxWidth: 200, boxHeight: 120 });
  assert.equal(viewerSafeResizeHandles(boxed).length, 8);
  assert.deepEqual(viewerSafeResizeHandles({ kind: "text", rotation: 0 }), ["tl", "tr", "bl", "br"]);
  // Rotation still withdraws every handle, box or not, because the drag maths assumes an upright box.
  assert.deepEqual(viewerSafeResizeHandles({ ...boxed, rotation: 30 }), []);
});

test("resizing a box moves the dragged edge and leaves the opposite one where it was", () => {
  const box = { width: 200, height: 100 };
  // Pulling the right edge 50px right widens by 50 and slides the centre by half of that.
  assert.deepEqual(viewerTextBoxResize("right", box, { x: 50, y: 999 }), { boxWidth: 250, boxHeight: 100, dx: 25, dy: 0 });
  // Pulling the top-left corner outwards grows both axes and moves the centre back towards it.
  assert.deepEqual(viewerTextBoxResize("tl", box, { x: -40, y: -20 }), { boxWidth: 240, boxHeight: 120, dx: -20, dy: -10 });
  assert.deepEqual(viewerTextBoxResize("bottom", box, { x: 999, y: 30 }), { boxWidth: 200, boxHeight: 130, dx: 0, dy: 15 });
  // A box cannot be collapsed to nothing: it stops at the size below which it stops being a box.
  assert.equal(viewerTextBoxResize("left", box, { x: 9999, y: 0 }).boxWidth, VIEWER_TEXT_MIN_BOX);
  assert.equal(viewerTextBoxResize("unknown", box, { x: 100, y: 100 }).boxWidth, 200);
});

test("the exported picture draws the lines the box shows, clipped to the box", () => {
  const context = recordingContext();
  drawTextLayer(context, normalizeTextLayer({ text: "hello world there", boxWidth: 124, boxHeight: 200 }, tenPixelsPerCharacter));
  assert.deepEqual(context.calls.filter(([name]) => name === "fillText").map(([, line]) => line), ["hello ", "world ", "there"]);
  assert.deepEqual(context.calls.find(([name]) => name === "rect"), ["rect", 0, 0, 124, 200]);
  assert.ok(context.calls.some(([name]) => name === "clip"), "a box shows only what fits in it, so the export must not spill the rest");
  // Text with no box is not clipped, and keeps drawing exactly the lines it was given.
  const unboxed = recordingContext();
  drawTextLayer(unboxed, normalizeTextLayer({ text: "one\ntwo" }, tenPixelsPerCharacter));
  assert.equal(unboxed.calls.some(([name]) => name === "clip"), false);
  assert.deepEqual(unboxed.calls.filter(([name]) => name === "fillText").map(([, line]) => line), ["one", "two"]);
});

test("the canvas wires a pointer-following ring, a crosshair text marquee, and the middle button", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  const css = fs.readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
  // The ring is a DOM circle, not a `cursor:` image: CSS cursors stop at 128px and a 300px brush
  // zoomed in needs far more. It is driven through a ref so a stroke does not repaint the app.
  assert.match(app, /viewerBrushCursorSize\(viewerBrush\.size, viewerZoom, at\.scale\)/);
  assert.match(app, /node\.style\.width = `\$\{diameter\}px`/);
  assert.match(app, /className=\{`viewer-brush-cursor \$\{viewerTool\} \$\{viewerPointerOverCanvas \? "" : "away"\}`\}/);
  assert.match(app, /onPointerMove=\{moveViewerImage\}/);
  assert.match(app, /const moveViewerImage = \(event\) => \{\s*moveViewerBrushCursor\(event\);/);
  for (const tool of ["brush", "eraser"]) assert.match(css, new RegExp(`\\.image-viewer-canvas\\.tool-${tool}[^\\n]*cursor: none;`));
  assert.match(css, /\.image-viewer-canvas\.tool-text[^\n]*cursor: crosshair;/);
  assert.match(css, /\.viewer-brush-cursor\.away \{ display: none; \}/);
  // The text tool sweeps a box and creates one only on release, so a drag is a box and a click is not.
  assert.match(app, /if \(viewerTool === "text"\) \{\s*startViewerTextMarquee\(event\);/);
  assert.match(app, /drag\?\.kind === "text-marquee"[\s\S]{0,160}createViewerText\(drag\.rect\)/);
  assert.match(app, /const boxed = rect\.width >= VIEWER_TEXT_MIN_BOX && rect\.height >= VIEWER_TEXT_MIN_BOX/);
  assert.match(app, /className="viewer-text-marquee"/);
  assert.match(css, /\.viewer-text-marquee \{[^}]*pointer-events: none;/);
  // A box is resized by the same handles a picture uses, but they change the box, not the glyphs.
  assert.match(app, /kind: "text-box"[\s\S]{0,320}pointerTarget: event\.currentTarget/);
  assert.match(app, /viewerTextBoxResize\(drag\.handle, drag\.box/);
  assert.match(app, /\(viewerTool === "move" \|\| \(viewerTool === "text" && textLayer\)\) && viewerLayerResizeEnabled/);
  // The middle button moves layers and pans whichever tool is selected, and the browser must not
  // claim it for autoscroll or for a primary-selection paste.
  assert.match(app, /const middleButton = event\.button === 1;/);
  assert.match(app, /if \(!middleButton && \(event\.button !== 0 \|\| event\.isPrimary === false\)\) return;/);
  assert.match(app, /if \(event\.button === 1\) \{\s*startViewerPan\(event\);/);
  assert.match(app, /const suppressViewerMiddleClick = \(event\) => \{\s*if \(event\.button === 1\) event\.preventDefault\(\);/);
  assert.match(app, /onMouseDown=\{suppressViewerMiddleClick\}/);
  assert.match(app, /onAuxClick=\{suppressViewerMiddleClick\}/);
  // The committed box shows the lines the export will draw, so neither wraps text the other way.
  assert.match(app, /\(layer\.lines \|\| \[layer\.text\]\)\.join\("\\n"\)/);
  assert.match(css, /\.viewer-text-content \{ white-space: pre;/);
});
