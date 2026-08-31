import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  boundedGalleryImageIndex,
  distributeGalleryCards,
  galleryImageSeed,
} from "../src/gallery-core.js";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

const pxFontRe = /(\d+(?:\.\d+)?)px/g;

function cssRules(css) {
  css = css.replace(/\/\*[\s\S]*?\*\//g, "");
  const rules = [];
  let i = 0;
  const n = css.length;
  while (i < n) {
    const open = css.indexOf("{", i);
    if (open === -1) break;
    const head = css.slice(i, open).trim();
    let j = open + 1;
    let depth = 1;
    while (j < n && depth > 0) {
      if (css[j] === "{") depth++;
      else if (css[j] === "}") depth--;
      j++;
    }
    const body = css.slice(open + 1, j - 1);
    if (head.startsWith("@")) {
      rules.push(...cssRules(body));
    } else {
      rules.push({ selectors: head, body });
    }
    i = j;
  }
  return rules;
}

function cssDeclarations(body) {
  const declarations = [];
  let i = 0;
  const n = body.length;
  while (i < n) {
    let j = i;
    let depth = 0;
    let quote = null;
    while (j < n) {
      const ch = body[j];
      if (quote) {
        if (ch === quote) quote = null;
      } else if (ch === '"' || ch === "'") {
        quote = ch;
      } else if (ch === "(") {
        depth++;
      } else if (ch === ")") {
        depth--;
      } else if (ch === ";" && depth === 0) {
        break;
      }
      j++;
    }
    const declaration = body.slice(i, j).trim();
    if (declaration) declarations.push(declaration);
    i = j + 1;
  }
  return declarations;
}

test("gallery image selection is valid on the first render after switching cards", () => {
  assert.equal(boundedGalleryImageIndex(2, 1), 0);
  assert.equal(boundedGalleryImageIndex(1, 3), 1);
  assert.equal(boundedGalleryImageIndex(-4, 3), 0);
  assert.equal(boundedGalleryImageIndex(Number.NaN, 3), 0);
  assert.equal(boundedGalleryImageIndex(7, 0), 0);
});

test("a per-image uint64 seed of zero never falls back to the card seed", () => {
  assert.equal(galleryImageSeed({ seed: "99", imageSeeds: ["0", "7"] }, 0), "0");
  assert.equal(galleryImageSeed({ seed: "99", imageSeeds: ["0", "7"] }, 1), "7");
  assert.equal(galleryImageSeed({ seed: "99", imageSeeds: [] }, 0), "99");
});

test("gallery masonry columns preserve row-first source distribution", () => {
  const cards = Array.from({ length: 9 }, (_, id) => ({ id }));
  assert.deepEqual(distributeGalleryCards(cards, 4).map((column) => column.map((card) => card.id)), [[0, 4, 8], [1, 5], [2, 6], [3, 7]]);
  assert.deepEqual(distributeGalleryCards(cards, 3).map((column) => column.map((card) => card.id)), [[0, 3, 6], [1, 4, 7], [2, 5, 8]]);
  assert.deepEqual(distributeGalleryCards(cards, 2).map((column) => column.map((card) => card.id)), [[0, 2, 4, 6, 8], [1, 3, 5, 7]]);
});

test("busy gallery dialogs cannot be dismissed behind an in-flight mutation", async () => {
  const [page, core, css] = await Promise.all([read("src/GalleryPage.jsx"), read("src/gallery-core.js"), read("src/styles.css")]);
  assert.match(core, /if \(canCloseRef\.current\) closeRef\.current\(\)/);
  assert.match(page, /useDialogLifecycle\(true, onClose, "", !busy\)/);
  assert.match(page, /useDialogLifecycle\(!loraOpen, onClose, "", !busy\)/);
  assert.match(page, /event\.target === event\.currentTarget && !busy && onClose\(\)/);
  assert.match(css, /\.gallery-focus > \.gallery-dialog-backdrop \{ z-index: 12; \}/,
    "the apply dialog must sit over the viewing room and its parameter drawer");
});

test("gallery room keeps the reference four-three-two column contract", async () => {
  const [page, css] = await Promise.all([read("src/GalleryPage.jsx"), read("src/styles.css")]);
  assert.match(page, /function GalleryCardColumns\(/);
  assert.match(page, /className="gallery-card-column"/);
  assert.match(page, /className="gallery-room-library"/);
  assert.match(page, /className="gallery-room-navigator"/);
  assert.match(css, /\.gallery-room \.gallery-card-grid \{[^}]*grid-template-columns: repeat\(4,minmax\(0,1fr\)\)/);
  assert.match(css, /@media \(max-width: 980px\)[\s\S]*?\.gallery-room \.gallery-card-grid \{ grid-template-columns: repeat\(3,minmax\(0,1fr\)\)/);
  assert.match(css, /@media \(max-width: 680px\)[\s\S]*?\.gallery-room \.gallery-card-grid \{ grid-template-columns: repeat\(2,minmax\(0,1fr\)\)/);
  assert.match(css, /\.gallery-room \.gallery-card-visual > img \{[^}]*height: auto/,
    "gallery covers must retain their intrinsic aspect ratio");
});

test("gallery opens directly on the themed archive without an editorial hero", async () => {
  const [page, css] = await Promise.all([read("src/GalleryPage.jsx"), read("src/styles.css")]);
  assert.doesNotMatch(page, /gallery-room-hero|gallery-room-showcase|showcaseCard/,
    "the removed giant title and random featured frame must not return");
  assert.match(page, /<FolderPlus size=\{14\} \/>新建收藏夹/,
    "creating a collection must remain available after removing the hero controls");
  assert.match(css, /\.gallery-room, \.gallery-focus \{[\s\S]*?--gallery-amber: var\(--lime\);/,
    "both the archive and its portalled viewer must inherit the current project accent");
  assert.match(css, /\.gallery-room-wash \{[^}]*rgb\(var\(--accent-rgb\) \/ \.12\)/,
    "the room atmosphere must derive from the active theme instead of a fixed warm glow");
});

test("the lightbox scales both portrait and panoramic originals fully inside the stage", async () => {
  const css = await read("src/styles.css");
  assert.match(css, /\.gallery-viewer-image-viewport \{[^}]*padding: clamp\(24px,4vw,72px\)/,
    "the opened image must have breathing room instead of filling the window edge to edge");
  assert.match(css, /\.gallery-viewer-image-viewport \.gallery-focus-frame > \.gallery-focus-full \{ width: auto; height: auto; max-width: 100%; max-height: 100%; \}/,
    "the decoded image must scale down against both axes while preserving its ratio");
  assert.match(css, /\.gallery-focus-shell \.gallery-viewer-image-viewport \{ background-color: transparent; background-image: none; \}/,
    "the high-specificity viewing stage must not restore an opaque black rectangle");
  assert.match(css, /\.gallery-viewer-image-viewport \.gallery-focus-frame \{[^}]*background: transparent;/,
    "the fitted image frame itself must remain transparent");
  assert.match(css, /@media \(max-width: 680px\)[\s\S]*?\.gallery-viewer-image-viewport \{ padding: 8px; \}/,
    "phone viewers must keep a compact fit margin");
});

test("every gallery UI font is at least 9px unless it is an explicit icon-hide zero", async () => {
  const css = await read("src/styles.css");
  const offenders = [];
  for (const rule of cssRules(css)) {
    if (!rule.selectors.includes("gallery")) continue;
    for (const declaration of cssDeclarations(rule.body)) {
      const separator = declaration.indexOf(":");
      if (separator === -1 || !/^(font-size|font)$/.test(declaration.slice(0, separator).trim())) continue;
      pxFontRe.lastIndex = 0;
      const pixel = pxFontRe.exec(declaration.slice(separator + 1));
      if (pixel) {
        const size = Number(pixel[1]);
        if (size !== 0 && size < 9) {
          offenders.push(`${rule.selectors} { ${declaration} }`);
        }
      }
    }
  }
  assert.deepEqual(offenders, [], `gallery font sizes must be 0 or >= 9px:\n${offenders.join("\n")}`);
});
