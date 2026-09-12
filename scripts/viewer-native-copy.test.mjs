import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// Over plain HTTP a page cannot write a picture to the clipboard: no clipboard API exists outside a
// secure context, so the viewer's "copy clean PNG" could only fail. The browser's own right-click
// menu can still copy -- it is how filecat's enlarged preview does it -- so on such a page the
// right-click is left to the browser, on the real <img>. These pin that, and that HTTPS is unchanged.

const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
const styles = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");

test("the mode is decided by the secure context, not by guessing at the address", () => {
  assert.match(app, /const nativeImageCopy = globalThis\.isSecureContext === false;/);
});

test("a plain right-click on a layer over HTTP reaches the browser's own menu", () => {
  const handlerStart = app.indexOf('onContextMenu={(event) => { event.stopPropagation(); setActiveViewerLayer(layer.id); if (nativeCopyLayer');
  assert.ok(handlerStart > 0, "the rendered layer must own the native-copy decision");
  const body = app.slice(handlerStart, handlerStart + 600);
  const bail = body.indexOf("if (nativeCopyLayer && !event.shiftKey) { hintNativeImageCopy(); return; }");
  const cancel = body.indexOf("event.preventDefault()");
  assert.ok(bail > 0 && cancel > bail, "preventDefault must come after the HTTP bail-out, or the browser menu never opens");
  assert.match(app.slice(app.indexOf("const nativeCopyLayer ="), app.indexOf("const edgeClass =", app.indexOf("const nativeCopyLayer ="))), /nativeImageCopy && !painted && normalizeRotation\(layer\.rotation\) === 0/,
    "only an ordinary, unrotated image may delegate copying to the native menu");
});

test("the picture itself takes the right-click only in that mode, so dragging is unchanged elsewhere", () => {
  // The browser offers "Copy image" only when the right-click lands on the <img>.
  assert.match(styles, /\.viewer-image-layer\.native-copy img \{ pointer-events: auto; -webkit-user-drag: none; \}/);
  assert.match(styles, /\.viewer-image-layer img \{[^}]*pointer-events: none;/, "the default must still let the pointer through");
  assert.match(app, /\$\{nativeCopyLayer \? "native-copy " : ""\}/);
  // Dragging survives an <img> target because the layer captures the pointer on itself.
  assert.match(app, /event\.currentTarget\.setPointerCapture\(event\.pointerId\);/);
});

test("Shift+right-click still opens the app menu, which points at the browser copy instead of failing", () => {
  assert.match(app, /nativeImageCopy && !hasLayerPaint\(viewerMenu\.layer\)/);
  assert.match(app, /复制请直接右键，用浏览器菜单「复制图片」/);
  // The delete action stays reachable: it is only in this menu, and there is no Delete key for layers.
  assert.match(app, /removeViewerLayer\(viewerMenu\.layer\.id\)/);
});

test("Ctrl+C over HTTP leaves the key to the browser and writes the layer marker in the copy event", () => {
  const keydown = app.slice(app.indexOf('if (event.key.toLowerCase() === "c" && activeViewerLayerItem && !activeCollage) {'));
  assert.ok(keydown.indexOf("if (nativeImageCopy) return;") < keydown.indexOf("event.preventDefault();"));
  assert.match(app, /event\.clipboardData\?\.setData\("text\/plain", `XIRAI_LAYER:\$\{activeViewerLayerItem\.id\}`\);/);
  assert.match(app, /window\.addEventListener\("copy", copyViewerLayerMarker\);/);
  assert.match(app, /return \(\) => window\.removeEventListener\("copy", copyViewerLayerMarker\);/);
  // Paste matches the marker first, so the copied layer wins over an old external picture.
  assert.match(app, /viewerClipboardPasteIntent\(\{ copied, marker, hasImages: files\.length > 0 \}\)/);
  assert.doesNotMatch(app.slice(app.indexOf("const pasteViewerLayer ="), app.indexOf("useEffect(() =>", app.indexOf("const pasteViewerLayer ="))), /Date\.now\(\) - copied\.copiedAt > 300000[\s\S]*cloneViewerLayer/,
    "external text may not fall back to an old in-memory layer");
});

test("over HTTPS or localhost the clean-PNG copy is exactly what it was", () => {
  assert.match(app, /await navigator\.clipboard\.write\(\[new ClipboardItem\(contents\)\]\);/);
  assert.match(app, /includeLayerMarker && !nativeImageCopy[\s\S]{0,180}viewerClipboard\.current = null/,
    "a failed secure-context copy must clear its internal marker");
  assert.match(app, /copyViewerLayer\(viewerMenu\.layer\)[\s\S]{0,100}复制实际 PNG/);
  assert.match(app, /hasLayerPaint\(layer\)[\s\S]{0,140}renderViewerLayerPng\(layer, \{ signal: token\.signal \}\)/,
    "paint and text copies must encode their rendered PNG instead of the untouched source");
});
