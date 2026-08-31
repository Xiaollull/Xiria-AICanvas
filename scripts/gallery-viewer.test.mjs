import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("gallery cards open an image-first viewer with an optional parameter drawer", async () => {
  const [page, css, core] = await Promise.all([read("src/GalleryPage.jsx"), read("src/styles.css"), read("src/gallery-core.js")]);

  assert.match(page, /function GalleryInspector\(/);
  assert.match(page, /function GalleryDetail\(/);
  assert.match(page, /useDialogLifecycle\(!applyOpen, closeViewer\)/,
    "the viewer must own Escape, focus return, and body scroll locking");
  assert.match(page, /className=\{`gallery-focus\$\{drawerOpen \? " inspector-open" : ""\}`\}/);
  assert.match(page, /aria-pressed=\{drawerOpen\}/);
  assert.match(page, /className="gallery-viewer-drawer" aria-hidden=\{!drawerOpen\}/);
  assert.match(page, /onClick=\{\(\) => setDrawerOpen\(\(current\) => !current\)\}/);
  assert.match(page, /inert=\{drawerOpen \? undefined : ""\}/,
    "a closed drawer must not remain in the tab order");
  assert.match(core, /!element\.closest\("\[inert\]"\)/,
    "dialog focus trapping must also skip an inert drawer subtree");
  const viewerSource = page.slice(page.indexOf("function GalleryDetail"), page.indexOf("export default function GalleryPage"));
  const drawerIndex = viewerSource.indexOf('className="gallery-viewer-drawer"');
  const applyDialogIndex = viewerSource.indexOf("{applyOpen && <ApplySettingsDialog");
  assert.ok(applyDialogIndex > viewerSource.indexOf("</section>", drawerIndex),
    "the full-screen apply dialog must not be trapped inside the transformed drawer");

  assert.match(page, /onWheel=\{zoomAtPointer\}/);
  assert.match(page, /onPointerDown=\{beginPan\}/);
  assert.match(page, /event\.key === "ArrowLeft"/);
  assert.match(page, /event\.key === "ArrowRight"/);
  assert.match(page, /ArrowLeft,\s*ArrowRight,/,
    "both navigation icons must be imported so entering the gallery cannot white-screen");
  assert.match(page, /onNavigate\(viewerCards\[\(cardIndex \+ direction \+ viewerCards\.length\) % viewerCards\.length\]\)/,
    "viewer arrows must browse cards in the active gallery scope");
  assert.match(page, /gallery-viewer-thumbs/);
  assert.match(page, /const viewerCards = selectedCard/,
    "the current collection must supply the viewer navigation scope");
  assert.match(page, /selectedCard && createPortal\(<GalleryDetail[\s\S]*?document\.body\)/,
    "opening a card must keep the collection mounted and portal the viewer above the app shell");
  assert.doesNotMatch(page, /selectedCard \? <GalleryDetail/,
    "the old replacement-detail layout must not return");

  const viewerCss = css.slice(css.indexOf("/* Gallery viewer"), css.indexOf("/* Dedicated LoRA asset intelligence page"));
  for (const selector of [
    ".gallery-focus-shell",
    ".gallery-viewer-image-viewport",
    ".gallery-viewer-drawer-sheet",
    ".gallery-focus.inspector-open .gallery-viewer-drawer-sheet",
    ".gallery-card-open-hint",
  ]) assert.ok(viewerCss.includes(`${selector} {`), `${selector} is required`);
  assert.match(viewerCss, /transform: translateX\(101%\)/,
    "the drawer must begin beyond the right edge");
  assert.match(viewerCss, /transform: translateX\(0\)/,
    "the drawer must slide into view");
  assert.doesNotMatch(viewerCss, /gallery-focus-frame\.ready \.gallery-focus-full \{[^}]*drop-shadow/,
    "large decoded originals must not use a per-pixel filter that can render as a black texture");
  assert.match(viewerCss, /gallery-viewer-image-viewport \.gallery-focus-frame > img \{[^}]*filter: none;/,
    "the viewing-room override must not reintroduce a GPU-heavy image filter");
  assert.match(viewerCss, /@media \(max-width: 720px\)[\s\S]*?\.gallery-viewer-drawer-sheet \{ width: min\(430px, calc\(100vw - 22px\)\); \}/,
    "the drawer must retain a usable width on phones");
});
