import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  HOVER_PREVIEW_DELAY,
  HOVER_PREVIEW_GAP,
  HOVER_PREVIEW_MARGIN,
  HOVER_PREVIEW_WIDTH,
  hasHoverPreview,
  hoverPreviewPlacement,
} from "../src/lora-hover-preview.js";
import {
  CARD_IMAGE_MAX_EDGE,
  encodeCardCanvas,
  prepareCardImage,
  scaledCardSize,
} from "../src/lora-card-image.js";
import { draftCommit, draftText, fromTextField } from "../src/lora-draft-field.js";

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
const viewport = { width: 1440, height: 900 };
const size = { width: HOVER_PREVIEW_WIDTH, height: 520 };
const row = (left, top, width = 320, height = 44) => ({ left, top, right: left + width, bottom: top + height });

test("the preview opens to the right of a row when there is room", () => {
  const placement = hoverPreviewPlacement({ anchor: row(40, 400), viewport, size });
  assert.equal(placement.side, "right");
  assert.equal(placement.left, 360 + HOVER_PREVIEW_GAP);
});

test("it flips to the left rather than hanging off the right edge", () => {
  const placement = hoverPreviewPlacement({ anchor: row(1060, 400), viewport, size });
  assert.equal(placement.side, "left");
  assert.equal(placement.left, 1060 - HOVER_PREVIEW_GAP - size.width);
  assert.ok(placement.left >= HOVER_PREVIEW_MARGIN);
});

test("when neither side fits it goes under the row rather than over it", () => {
  // A narrow window: 220px of row with 250px to its right and 230px to its left,
  // and a 300px panel. Squeezing it in beside the row would put the picture on
  // top of the thing the pointer is resting on.
  const anchor = row(230, 300, 220);
  const placement = hoverPreviewPlacement({ anchor, viewport: { width: 700, height: 900 }, size });
  assert.equal(placement.side, "below");
  assert.ok(placement.top >= anchor.bottom, "the panel starts below the row");
  assert.ok(placement.left >= HOVER_PREVIEW_MARGIN);
  assert.ok(placement.left + size.width <= 700 - HOVER_PREVIEW_MARGIN);
});

test("it goes above the row instead when the space below is smaller", () => {
  const anchor = row(230, 700, 220);
  const placement = hoverPreviewPlacement({ anchor, viewport: { width: 700, height: 900 }, size });
  assert.equal(placement.side, "above");
  assert.equal(placement.top + size.height, anchor.top - HOVER_PREVIEW_GAP);
});

test("a side placement is preferred whenever one actually fits", () => {
  const roomyRight = hoverPreviewPlacement({ anchor: row(40, 400), viewport, size });
  assert.equal(roomyRight.side, "right");
  // Only the left has 300px here, so it is taken rather than falling to vertical.
  const onlyLeft = hoverPreviewPlacement({ anchor: row(340, 400, 220), viewport: { width: 700, height: 900 }, size });
  assert.equal(onlyLeft.side, "left");
  assert.ok(onlyLeft.left + size.width <= 340 - HOVER_PREVIEW_GAP);
});

test("it is centred on the row it belongs to, then clamped to the viewport", () => {
  const middle = hoverPreviewPlacement({ anchor: row(40, 400), viewport, size });
  assert.equal(middle.top, 400 + 22 - 260);

  const nearTop = hoverPreviewPlacement({ anchor: row(40, 10), viewport, size });
  assert.equal(nearTop.top, HOVER_PREVIEW_MARGIN);

  const nearBottom = hoverPreviewPlacement({ anchor: row(40, 870), viewport, size });
  assert.equal(nearBottom.top, viewport.height - size.height - HOVER_PREVIEW_MARGIN);
});

test("a viewport shorter than the panel pins it to the top instead of producing a negative offset", () => {
  const placement = hoverPreviewPlacement({ anchor: row(40, 100), viewport: { width: 1440, height: 380 }, size });
  assert.equal(placement.top, HOVER_PREVIEW_MARGIN);
  assert.ok(Number.isFinite(placement.left));
});

test("missing measurements degrade to a placed panel rather than NaN", () => {
  const placement = hoverPreviewPlacement();
  assert.equal(placement.left, HOVER_PREVIEW_MARGIN);
  assert.equal(placement.top, HOVER_PREVIEW_MARGIN);
});

test("a LoRA nobody has customised still opens a card, and says what is missing", () => {
  // Requiring a cover, prompt, note or tag meant hovering an untouched LoRA did
  // nothing at all — indistinguishable from the feature being broken, and with
  // no hint that there was a card to fill in.
  assert.equal(hasHoverPreview({ title: "a.safetensors", fileName: "a.safetensors", coverUrl: "", prompt: "", note: "", tags: [] }), true);
  assert.equal(hasHoverPreview({ coverUrl: "/api/lora-cards/asset?id=x" }), true);
  // Nothing at all is still nothing: a row with no identity opens no panel.
  assert.equal(hasHoverPreview(null), false);
  assert.equal(hasHoverPreview({}), false);
});

test("the panel is placed synchronously and drawn outside the page flow", async () => {
  const source = await readSource("src/LoraHoverPreview.jsx");
  // It used to measure itself inside `requestAnimationFrame`, which a throttled
  // or occluded page never delivers — and until it arrived the panel stayed
  // `visibility: hidden`, so hovering appeared to do nothing.
  assert.match(source, /useLayoutEffect\(\(\) => \{/);
  assert.doesNotMatch(source, /window\.requestAnimationFrame\(/);
  // The cover arrives after the first measurement and changes the height.
  assert.match(source, /onLoad=\{remeasure\}/);
  const styles = await readSource("src/styles.css");
  assert.match(styles, /\.lora-hover-preview \{[^}]*position: fixed/);
  assert.match(styles, /\.lora-hover-preview \{[^}]*pointer-events: none/);
});

test("the card editor is reachable without having to guess where it is", async () => {
  const [styles, app, page, group] = await Promise.all([
    readSource("src/styles.css"),
    readSource("src/App.jsx"),
    readSource("src/LoraManagerPage.jsx"),
    readSource("src/LoraGroupPanel.jsx"),
  ]);
  // The button on a library card was `opacity: 0` until the card was hovered,
  // which hid the whole feature from anyone not already looking for it.
  const customize = styles.slice(styles.indexOf(".lora-card-customize {"));
  assert.doesNotMatch(customize.slice(0, customize.indexOf("}")), /opacity: 0/);
  for (const source of [app, page]) assert.match(source, /<Palette size=\{13\} \/>自定义/);
  // And the group's effect images needed the body expanded first, so the cover
  // could not be found at all from the group list.
  assert.match(group, /className=\{`lora-group-artwork-open/);
  assert.match(group, /onClick=\{\(\) => setExpanded\(group\.id\)\}/);
});

test("the open delay is long enough to run a pointer down a list without strobing", () => {
  assert.ok(HOVER_PREVIEW_DELAY >= 120 && HOVER_PREVIEW_DELAY <= 400);
});

test("the preview is drawn at a size worth stopping for", async () => {
  assert.ok(HOVER_PREVIEW_WIDTH >= 260);
  const styles = await readSource("src/styles.css");
  assert.match(styles, /\.lora-hover-preview \{[^}]*width: 300px/);
  // Rows live inside scrolling panels and a modal, so the panel is drawn into
  // <body> and positioned in viewport coordinates.
  assert.match(styles, /\.lora-hover-preview \{[^}]*position: fixed/);
  assert.match(await readSource("src/LoraHoverPreview.jsx"), /createPortal\(/);
});

test("a picture is scaled by its longest edge, and a small one is left alone", () => {
  assert.deepEqual(scaledCardSize(1920, 1080, 1024), { width: 1024, height: 576, scaled: true });
  assert.deepEqual(scaledCardSize(1080, 1920, 1024), { width: 576, height: 1024, scaled: true });
  assert.deepEqual(scaledCardSize(800, 600, 1024), { width: 800, height: 600, scaled: false });
  // A pathologically thin source must still have a pixel in each direction.
  assert.deepEqual(scaledCardSize(4000, 3, 1024), { width: 1024, height: 1, scaled: true });
});

test("the encoder trusts what the canvas returns, not what it was asked for", () => {
  // `toDataURL` answers with PNG when it does not know the requested type rather
  // than failing, so the prefix decides whether a format actually worked.
  const webpless = { toDataURL: (type) => type === "image/jpeg" ? "data:image/jpeg;base64,AAAA" : "data:image/png;base64,AAAA" };
  assert.deepEqual(encodeCardCanvas(webpless, 0.86), { dataUrl: "data:image/jpeg;base64,AAAA", contentType: "image/jpeg" });
  const modern = { toDataURL: () => "data:image/webp;base64,AAAA" };
  assert.equal(encodeCardCanvas(modern, 0.86).contentType, "image/webp");
  assert.throws(() => encodeCardCanvas({ toDataURL: () => "data:image/png;base64,AAAA" }, 0.86), /无法编码/);
});

test("preparing an image downscales it and reports what it produced", async () => {
  const drawn = [];
  const canvas = {
    getContext: () => ({ fillStyle: "", fillRect: (...args) => drawn.push(["fill", ...args]), drawImage: (...args) => drawn.push(["draw", args.length]) }),
    toDataURL: () => "data:image/webp;base64,AAAA",
  };
  const result = await prepareCardImage("/api/inference/history/assets/x", {
    maxEdge: CARD_IMAGE_MAX_EDGE,
    loadImage: async () => ({ naturalWidth: 2048, naturalHeight: 1024 }),
    createCanvas: (width, height) => ({ ...canvas, width, height }),
  });
  assert.deepEqual(result, { dataUrl: "data:image/webp;base64,AAAA", contentType: "image/webp", width: 1024, height: 512, scaled: true });
  // The white fill is what keeps a transparent source from compositing onto
  // black if the browser falls back to JPEG.
  assert.deepEqual(drawn[0], ["fill", 0, 0, 1024, 512]);
});

test("an unreadable image is an error rather than a zero-sized card", async () => {
  await assert.rejects(
    () => prepareCardImage("x", { loadImage: async () => ({ naturalWidth: 0, naturalHeight: 0 }) }),
    /尺寸无法识别/,
  );
});

test("a text box shows its draft, and commits only a real change", () => {
  // Without a draft the box would show the stored value, which every one of these
  // stores has already trimmed — so the space just typed is gone before the next
  // render and the second word can never be started.
  assert.equal(draftText("summer", null), "summer");
  assert.equal(draftText("summer", "summer, "), "summer, ");
  assert.equal(draftText("summer", ""), "", "clearing the box is a draft, not an absent one");

  assert.deepEqual(draftCommit("summer", "summer, "), { text: "summer, ", changed: true });
  // Opening a dialog and closing it again must not mark the store dirty.
  assert.deepEqual(draftCommit("summer", null), { text: "summer", changed: false });
  assert.deepEqual(draftCommit("summer", "summer"), { text: "summer", changed: false });
});

test("the Escape guard survives a key event whose target is not an element", () => {
  // It runs from a capture listener that sees every key event, including ones
  // targeting the document.
  assert.equal(fromTextField(null), false);
  assert.equal(fromTextField({}), false);
  assert.equal(fromTextField({ target: {} }), false);
  assert.equal(fromTextField({ target: { closest: (selector) => selector.includes("textarea") ? {} : null } }), true);
  assert.equal(fromTextField({ target: { closest: () => null } }), false);
});

test("every box writing into a trimming store uses the shared draft field", async () => {
  const [mount, group, editor] = await Promise.all([
    readSource("src/LoraMountPanel.jsx"),
    readSource("src/LoraGroupPanel.jsx"),
    readSource("src/LoraCardEditor.jsx"),
  ]);
  for (const [name, source] of [["LoraMountPanel", mount], ["LoraGroupPanel", group], ["LoraCardEditor", editor]]) {
    assert.match(source, /useDraftField\(/, `${name} keeps a draft while a box is focused`);
    // `presetPrompt` and every card field are trimmed on the way in, so writing
    // each keystroke through would swallow a trailing space and stall the caret.
    assert.doesNotMatch(source, /onChange=\{\(event\) => \w+\(group\.id, \{ (?:presetPrompt|note): event\.target\.value \}\)\}/, `${name} does not write raw keystrokes into a trimming store`);
  }
  // Escape inside a text box reverts that box; the dialog only closes when the
  // key did not come from one.
  assert.match(editor, /if \(pickingRef\.current \|\| fromTextField\(event\)\) return;/);
});
