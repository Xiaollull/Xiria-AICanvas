import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { viewerBrushCursorSize, viewerMarqueeRect, VIEWER_BRUSH_CURSOR_MAX_PX, VIEWER_BRUSH_CURSOR_MIN_PX } from "../src/viewer-geometry.js";
import {
  drawTextLayer,
  normalizeManualLayout,
  normalizeTextLayer,
  serializeViewerLayer,
  viewerPaintOrder,
  viewerSafeResizeHandles,
  viewerTextBox,
  viewerTextBoxResize,
  viewerTextStyleFor,
  wrapTextLines,
  VIEWER_TEXT_MIN_BOX,
  VIEWER_TEXT_PADDING,
  VIEWER_TEXT_STYLES,
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
  // The text tool sweeps a box and creates one only on release; a click that swept nothing leaves
  // nothing behind but a cleared selection.
  assert.match(app, /if \(viewerTool === "text"\) \{\s*startViewerTextMarquee\(event\);/);
  assert.match(app, /drag\?\.kind === "text-marquee"[\s\S]{0,160}createViewerText\(drag\.rect\)/);
  assert.match(app, /if \(rect\.width < VIEWER_TEXT_MIN_BOX \|\| rect\.height < VIEWER_TEXT_MIN_BOX\) \{\s*setActiveViewerLayer\(""\);\s*return;/);
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

test("text sits above pictures on the canvas and in the exported picture alike", () => {
  const image = (id) => ({ id, kind: "image", naturalWidth: 10, naturalHeight: 10 });
  const text = (id) => ({ id, kind: "text", text: id });
  // Whatever order the layers arrived in, the text is painted last.
  assert.deepEqual(viewerPaintOrder([text("t1"), image("i1"), text("t2"), image("i2")]).map((layer) => layer.id), ["i1", "i2", "t1", "t2"]);
  // Each kind keeps its own stacking, so an image raised above another stays above it.
  assert.deepEqual(viewerPaintOrder([image("i2"), image("i1")]).map((layer) => layer.id), ["i2", "i1"]);
  assert.deepEqual(viewerPaintOrder([]), []);
  assert.deepEqual(viewerPaintOrder(null), []);

  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  const css = fs.readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
  // Both renderers take the same order from the same helper, so neither can stack them differently.
  assert.match(app, /viewerPaintOrder\(viewerLayers\)\.map\(\(layer\) => \{/);
  assert.match(app, /const sourceLayers = viewerPaintOrder\(cloneViewerLayers\(viewerLayers\)\);/);
  // On screen the stacking is the stylesheet's: above a plain image (3) and above a selected one (8).
  assert.match(css, /\.viewer-text-layer \{ z-index: 6; \}/);
  assert.match(css, /\.viewer-text-layer\.active \{ z-index: 9; \}/);
});

test("text written on a picture travels with it, and the picture never travels with the text", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  // The binding points one way: a text layer names the picture it was written on, never the reverse.
  assert.match(app, /function viewerMovesWithLayer\(layer, movedId\) \{\s*return layer\?\.id === movedId \|\| \(viewerLayerKind\(layer\) === "text" && Boolean\(layer\?\.attachedTo\) && layer\.attachedTo === movedId\);/);
  assert.match(app, /attachedTo: viewerImageUnderPoint\(\{ x: rect\.centerX, y: rect\.centerY \}\)/);
  // A drag replays from the positions held at the start, so samples cannot accumulate drift.
  assert.match(app, /attached: viewerLayers\.filter\(\(item\) => item\.id !== layer\.id && viewerMovesWithLayer\(item, layer\.id\)\)\.map/);
  assert.match(app, /origin \? \{ \.\.\.layer, x: origin\.x \+ x - drag\.layerX, y: origin\.y \+ y - drag\.layerY \} : layer/);
  // The arrow keys move a picture the same way the pointer does.
  assert.match(app, /viewerMovesWithLayer\(layer, activeViewerLayerItem\.id\)\s*\?\s*\{ \.\.\.layer, x: layer\.x \+ adjustment\[0\], y: layer\.y \+ adjustment\[1\] \}/);
});

test("a finished box stops framing itself once attention moves elsewhere, and reads on any backdrop", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  const css = fs.readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
  // Pressing bare canvas clears the selection; pressing another layer moves it there already.
  assert.match(app, /const startViewerDrag = \(event\) => \{[\s\S]{0,400}setActiveViewerLayer\(""\);/);
  // A lavender band between two dark ones stays visible over a pale picture and a dark one alike.
  assert.match(css, /\.viewer-text-editor \{[^}]*box-shadow: 0 0 0 1px rgba\(9,10,12,\.85\), 0 0 0 3px #c8acfb, 0 0 0 4px rgba\(9,10,12,\.65\);/);
  assert.match(css, /\.viewer-text-layer\.active \{ outline: 0; box-shadow: 0 0 0 1px rgba\(9,10,12,\.85\), 0 0 0 2px #c8acfb, 0 0 0 3px rgba\(9,10,12,\.6\); \}/);
  // Nothing is painted behind the text being typed, so the box reads as the picture will export it.
  assert.match(css, /html\[data-theme-mode="light"\] \.viewer-text-editor \{ color: inherit; background: transparent; \}/);
  // A text box's grips are square and smaller than a picture's round ones (20px hit, 13px visual).
  assert.match(css, /\.viewer-image-layer\.viewer-text-layer \.layer-corner \{ width: 13px; height: 13px; border-radius: 0;/);
  assert.match(css, /\.viewer-image-layer\.viewer-text-layer \.layer-corner::after \{ width: 7px; height: 7px; border-radius: 0;/);
});

test("the font list previews every family it offers, and keeps a family it does not know", () => {
  assert.ok(VIEWER_TEXT_STYLES.length >= 8, "the list has to be worth opening");
  // Every entry names a family that ships with Windows, and the CJK ones are named as the user does.
  for (const style of VIEWER_TEXT_STYLES) {
    assert.match(style.fontFamily, /,|^[A-Za-z]/, `${style.id} must name a real family stack`);
    assert.ok(style.label.length > 0 && style.label.length <= 12, `${style.id} label must fit the toolbar`);
  }
  assert.equal(new Set(VIEWER_TEXT_STYLES.map((style) => style.id)).size, VIEWER_TEXT_STYLES.length);
  assert.equal(viewerTextStyleFor("Arial, Helvetica, sans-serif").id, "sans");
  assert.equal(viewerTextStyleFor("arial, helvetica, sans-serif").id, "sans", "matching must not turn on case");
  // A layout saved before this list existed keeps the family it was written with.
  const unknown = viewerTextStyleFor("Papyrus, fantasy");
  assert.deepEqual([unknown.id, unknown.fontFamily], ["custom", "Papyrus, fantasy"]);
  assert.equal(viewerTextStyleFor("").id, "custom");

  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  const css = fs.readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
  // Each row is set in the family it names: the list is the preview.
  assert.match(app, /VIEWER_TEXT_STYLES\.map\(\(style\) => <option key=\{style\.id\} value=\{style\.id\} style=\{\{ fontFamily: style\.fontFamily \}\}>\{style\.label\}<\/option>\)/);
  assert.match(app, /updateViewerTextProperties\(activeViewerTextLayer, \{ fontFamily: style\.fontFamily, font: style\.fontFamily \}\)/);
  assert.match(css, /\.viewer-font-select select option \{[^}]*font-size: 13px;/);
});
