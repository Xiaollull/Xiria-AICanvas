import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { viewerBrushCursorSize, VIEWER_BRUSH_CURSOR_MAX_PX, VIEWER_BRUSH_CURSOR_MIN_PX } from "../src/viewer-geometry.js";

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

test("the canvas wires a pointer-following ring and the middle button, and offers no text tool", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  const css = fs.readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
  // The ring is a DOM circle, not a `cursor:` image: CSS cursors stop at 128px and a 300px brush
  // zoomed in needs far more. It is driven through a ref so a stroke does not repaint the app.
  assert.match(app, /viewerBrushCursorSize\(viewerBrush\.size, viewerZoom, at\.scale\)/);
  assert.match(app, /node\.style\.width = `\$\{diameter\}px`/);
  assert.match(app, /className=\{`viewer-brush-cursor \$\{viewerTool\} \$\{viewerPointerOverCanvas \? "" : "away"\}`\}/);
  assert.match(app, /const moveViewerImage = \(event\) => \{\s*moveViewerBrushCursor\(event\);/);
  for (const tool of ["brush", "eraser"]) assert.match(css, new RegExp(`\\.image-viewer-canvas\\.tool-${tool}[^\\n]*cursor: none;`));
  assert.match(css, /\.viewer-brush-cursor\.away \{ display: none; \}/);
  // The middle button moves layers and pans whichever tool is selected, and the browser must not
  // claim it for autoscroll or for a primary-selection paste.
  assert.match(app, /const middleButton = event\.button === 1;/);
  assert.match(app, /if \(!middleButton && \(event\.button !== 0 \|\| event\.isPrimary === false\)\) return;/);
  assert.match(app, /if \(event\.button === 1\) \{\s*startViewerPan\(event\);/);
  assert.match(app, /const suppressViewerMiddleClick = \(event\) => \{\s*if \(event\.button === 1\) event\.preventDefault\(\);/);
  assert.match(app, /onMouseDown=\{suppressViewerMiddleClick\}/);
  assert.match(app, /onAuxClick=\{suppressViewerMiddleClick\}/);
  // Pressing bare canvas still drops the selection, so a picture stops framing itself.
  assert.match(app, /const startViewerDrag = \(event\) => \{[\s\S]{0,400}setActiveViewerLayer\(""\);/);
  // No text tool remains to sweep a box or to write into one.
  assert.doesNotMatch(app, /startViewerTextMarquee|viewer-text-marquee|text-box/);
  assert.doesNotMatch(css, /viewer-text-marquee|tool-text/);
});
